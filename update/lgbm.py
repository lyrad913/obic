"""
LightGBM 단독 모델 (수정 버전)
주요 수정사항:
1. 시간대 처리 로직 개선 (KST 이중 변환 방지)
2. 물리적 제약 조건 완화
3. 디버깅 로그 추가
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from .config import TrainingConfig
from .data import DataBundle, ensure_kst_time, prepare_data
from .utils import set_seed


def prepare_data_no_scaling(config: TrainingConfig) -> DataBundle:
    """스케일링 없는 데이터 준비"""
    from .data import (
        preprocess_frames,
        select_numerical_columns,
        split_train_validation,
        to_numpy,
    )
    
    train_df, test_df, submission_df = preprocess_frames(config)
    feature_exclude = {config.target_col, config.group_col, config.time_col, "type"}
    feature_columns = select_numerical_columns(train_df, feature_exclude)
    feature_columns = sorted(feature_columns)

    train_split, val_split = split_train_validation(train_df, config)
    train_groups = train_split.select(config.group_col).n_unique()
    val_groups = val_split.select(config.group_col).n_unique()
    logger.info(
        f"검증 세트 분할 완료: train_groups={train_groups}, val_groups={val_groups}, "
        f"train_rows={train_split.height}, val_rows={val_split.height}"
    )

    logger.info("스케일링 없이 데이터 준비 중... (트리 모델용)")
    x_train = to_numpy(train_split, feature_columns)
    x_val = to_numpy(val_split, feature_columns)
    x_test = to_numpy(test_df, feature_columns)

    y_train = train_split.select(config.target_col).to_numpy().astype(np.float32).ravel()
    y_val = val_split.select(config.target_col).to_numpy().astype(np.float32).ravel()

    train_meta = train_split.select([config.group_col, config.time_col])
    val_meta = val_split.select([config.group_col, config.time_col])
    meta_cols = [config.time_col, config.group_col]
    if "type" in test_df.columns:
        meta_cols.append("type")
    test_meta = test_df.select(meta_cols)
    rename_map = {}
    if config.time_col != "time":
        rename_map[config.time_col] = "time"
    if config.group_col != "pv_id":
        rename_map[config.group_col] = "pv_id"
    if rename_map:
        test_meta = test_meta.rename(rename_map)
    if "type" not in test_meta.columns:
        test_meta = test_meta.with_columns(pl.lit("test").alias("type"))
    submission_meta = submission_df.select(["time", "pv_id", "type"])

    return DataBundle(
        feature_names=feature_columns,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        submission_template=submission_meta,
        scaler=None,
    )


class LightGBMSingle:
    """LightGBM 단독 모델"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = self._init_model()
        self.checkpoint_dir = self.config.checkpoint_dir
        
    def _init_model(self) -> LGBMRegressor:
        """LightGBM 초기화"""
        use_gpu = self.config.use_gpu
        
        lgbm_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 8,
            "num_leaves": 63,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": self.config.seed,
            "device": "gpu" if use_gpu else "cpu",
            "verbose": -1,
            "objective": "mae",
            "metric": "mae",
        }
        
        return LGBMRegressor(**lgbm_params)
    
    def _save_checkpoint(self, filename: str) -> None:
        """체크포인트 저장"""
        booster = getattr(self.model, "booster_", None)
        if booster is None:
            logger.warning("LightGBM booster가 없어 체크포인트를 건너뜁니다.")
            return
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(checkpoint_path))
        logger.info(f"💾 모델 체크포인트 저장: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path, feature_count: int) -> bool:
        """체크포인트 로드"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.warning(f"지정한 체크포인트가 존재하지 않습니다: {checkpoint_path}")
            return False

        booster = lgb.Booster(model_file=str(checkpoint_path))
        self.model._Booster = booster
        self.model.fitted_ = True
        self.model._n_features_in_ = feature_count
        self.model.n_features_in_ = feature_count
        self.model._n_features = feature_count

        best_iteration = booster.best_iteration
        if best_iteration is not None:
            self.model._best_iteration = best_iteration

        logger.info(f"✅ 체크포인트 불러오기 완료: {checkpoint_path}")
        return True
    
    def fit_stacking(self, bundle: DataBundle) -> None:
        """GroupKFold OOF 학습"""
        logger.info("🌳 [LightGBM Single] GroupKFold 학습 시작")
        
        logger.info("1단계: OOF(Out-of-Fold) 예측 생성 중...")
        
        oof_train = np.zeros(len(bundle.x_train))
        oof_val = np.zeros(len(bundle.x_val))
        
        gkf = GroupKFold(n_splits=5)
        train_groups = bundle.train_meta.select(self.config.group_col).to_numpy().ravel()
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(bundle.x_train, bundle.y_train, groups=train_groups)):
            logger.info(f"  Fold {fold+1}/5 학습 중...")
            
            X_fold_train = bundle.x_train[train_idx]
            y_fold_train = bundle.y_train[train_idx]
            X_fold_val = bundle.x_train[val_idx]
            y_fold_val = bundle.y_train[val_idx]
            
            self.model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
            )
            self._save_checkpoint(f"lightgbm_single_fold{fold+1}.txt")
            
            oof_train[val_idx] = self.model.predict(X_fold_val).clip(min=0)
            
            fold_mae = mean_absolute_error(y_fold_val, oof_train[val_idx])
            logger.info(f"    Fold {fold+1} MAE: {fold_mae:.4f}")
        
        oof_mae = mean_absolute_error(bundle.y_train, oof_train)
        logger.info(f"  전체 OOF MAE: {oof_mae:.4f}")
        
        logger.info("2단계: 전체 데이터로 최종 모델 재학습 중...")
        
        self.model = self._init_model()
        self.model.fit(
            bundle.x_train,
            bundle.y_train,
            eval_set=[(bundle.x_val, bundle.y_val)],
        )
        self._save_checkpoint("lightgbm_single_final.txt")
        
        oof_val = self.model.predict(bundle.x_val).clip(min=0)
        val_mae = mean_absolute_error(bundle.y_val, oof_val)
        
        logger.info(f"🎯 Validation MAE: {val_mae:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Test 예측"""
        return self.model.predict(X_test).clip(min=0)


def _apply_physical_constraints(predictions: np.ndarray, test_meta: pl.DataFrame) -> np.ndarray:
    df = test_meta.with_columns(
        pl.Series("nins", np.asarray(predictions, dtype=np.float32))
    )

    # UTC/KST 정규화
    df = ensure_kst_time(df, "time")
    
    # ⭐ KST 기준으로 hour와 month 추출
    df = df.with_columns([
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.month().alias("month"),
    ])

    # ✅ 야간(5~20시 외) 일사량 0 (여름 일몰 고려)
    df = df.with_columns([
        pl.when(pl.col("hour").is_between(5, 20, closed="both"))
        .then(pl.col("nins"))
        .otherwise(0.0)
        .alias("nins")
    ])

    # ✅ 계절별 상한선
    df = df.with_columns([
        pl.when(pl.col("month").is_in([12, 1, 2]))
        .then(pl.col("nins").clip(0, 900))
        .when(pl.col("month").is_in([6, 7, 8]))
        .then(pl.col("nins").clip(0, 1200))
        .otherwise(pl.col("nins").clip(0, 1100))
        .alias("nins")
    ])

    return df["nins"].to_numpy().astype(np.float32, copy=False)


def run_lightgbm_single_pipeline(
    config: TrainingConfig,
    resume_checkpoint: Path | None = None,
    resume_latest: bool = False,
):
    """LightGBM 단독 파이프라인 실행"""
    config.resolve_paths()
    checkpoint_path: Path | None = None
    if resume_checkpoint is not None:
        checkpoint_path = Path(resume_checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    if checkpoint_path is None and resume_latest:
        candidate = config.checkpoint_dir / "lightgbm_single_final.txt"
        if candidate.exists():
            checkpoint_path = candidate
        else:
            logger.warning(f"최신 체크포인트를 찾을 수 없습니다: {candidate}")
    
    set_seed(config.seed)
    
    # 스케일링 없는 데이터 준비
    bundle = prepare_data_no_scaling(config)
    
    logger.info("LightGBM Single 학습 시작")
    
    # 학습 또는 체크포인트 복구
    model = LightGBMSingle(config)
    checkpoint_loaded = False
    if checkpoint_path is not None:
        checkpoint_loaded = model.load_checkpoint(checkpoint_path, len(bundle.feature_names))
        if not checkpoint_loaded:
            logger.warning("체크포인트 로드에 실패하여 새로 학습을 진행합니다.")
    if not checkpoint_loaded:
        model.fit_stacking(bundle)
    
    # 예측
    logger.info("🔮 Test 데이터 예측 중...")
    predictions = model.predict(bundle.x_test)
    
    logger.info(f"📊 예측 통계 (제약 전):")
    logger.info(f"  - 평균: {predictions.mean():.2f}")
    logger.info(f"  - 최대: {predictions.max():.2f}")
    logger.info(f"  - 최소: {predictions.min():.2f}")
    
    # 🔍 시간 정보 디버깅
    logger.info(f"📅 Test 데이터 시간 정보:")
    logger.info(f"  - 첫 5개 시간: {bundle.test_meta['time'].head()}")
    
    # 물리적 제약 후처리
    predictions = _apply_physical_constraints(predictions, bundle.test_meta)
    
    # 제출 파일 저장
    from .pipeline import _save_submission
    submission_path = _save_submission(predictions, bundle, config)
    
    logger.info(f"✅ 제출 파일 저장 완료: {submission_path}")
    
    return {
        'method': 'lightgbm_single',
        'submission_path': str(submission_path),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="LightGBM 단독 학습")
    parser.add_argument('--data-dir', type=Path, default=Path('./data'))
    parser.add_argument('--submission-name', type=str, default='lightgbm_single_submission.csv')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume-checkpoint', type=Path, default=None)
    parser.add_argument('--resume-latest', action='store_true')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        submission_name=args.submission_name,
        use_gpu=not args.cpu,
        seed=args.seed,
    )
    
    result = run_lightgbm_single_pipeline(
        config,
        resume_checkpoint=args.resume_checkpoint,
        resume_latest=args.resume_latest,
    )
    print(f"\n🎉 완료! 결과: {result['submission_path']}")
