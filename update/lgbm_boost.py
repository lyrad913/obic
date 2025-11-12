"""
LightGBM Weighted Boosting (가중치 앙상블)
- 각 fold의 OOF MAE 성능에 따라 가중치 부여
- 성능 좋은 모델에 더 높은 가중치
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
from .data import DataBundle, ensure_kst_time
from .utils import set_seed


class LightGBMWeightedBoosting:
    """성능 기반 가중치 부여 LightGBM 앙상블"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.fold_models = []
        self.fold_weights = []
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
    
    def _calculate_weights(self, fold_maes: list[float], method: str = "inverse") -> np.ndarray:
        """
        각 fold의 MAE를 기반으로 가중치 계산
        
        Args:
            fold_maes: 각 fold의 MAE 리스트
            method: 가중치 계산 방법
                - "inverse": 1/MAE (역수)
                - "softmax": softmax(-MAE)
                - "rank": 순위 기반
        """
        fold_maes = np.array(fold_maes)
        
        if method == "inverse":
            # 역수 방식: MAE가 낮을수록 가중치 높음
            weights = 1.0 / fold_maes
            weights = weights / weights.sum()
            
        elif method == "softmax":
            # Softmax 방식: 성능 차이를 exponential로 증폭
            scores = -fold_maes  # 낮을수록 좋으므로 음수
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)  # 정규화
            exp_scores = np.exp(scores)
            weights = exp_scores / exp_scores.sum()
            
        elif method == "rank":
            # 순위 기반: 1등에게 가장 높은 가중치
            ranks = np.argsort(np.argsort(fold_maes)) + 1  # 1부터 시작
            weights = 1.0 / ranks
            weights = weights / weights.sum()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return weights
    
    def fit_weighted_boosting(
        self, 
        bundle: DataBundle, 
        weight_method: str = "inverse",
        n_splits: int = 5
    ) -> None:
        """
        GroupKFold로 학습하고 성능 기반 가중치 계산
        
        Args:
            bundle: 데이터 번들
            weight_method: 가중치 계산 방법 ("inverse", "softmax", "rank")
            n_splits: Fold 개수
        """
        logger.info(f"🌳 [LightGBM Weighted Boosting] GroupKFold 학습 시작 (method={weight_method})")
        
        fold_maes = []
        gkf = GroupKFold(n_splits=n_splits)
        train_groups = bundle.train_meta.select(self.config.group_col).to_numpy().ravel()
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(bundle.x_train, bundle.y_train, groups=train_groups)):
            logger.info(f"  Fold {fold+1}/{n_splits} 학습 중...")
            
            X_fold_train = bundle.x_train[train_idx]
            y_fold_train = bundle.y_train[train_idx]
            X_fold_val = bundle.x_train[val_idx]
            y_fold_val = bundle.y_train[val_idx]
            
            model = self._init_model()
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
            )
            
            # Fold 모델 저장
            self.fold_models.append(model)
            
            # OOF 성능 측정
            oof_pred = model.predict(X_fold_val).clip(min=0)
            fold_mae = mean_absolute_error(y_fold_val, oof_pred)
            fold_maes.append(fold_mae)
            
            logger.info(f"    Fold {fold+1} MAE: {fold_mae:.4f}")
            
            # 체크포인트 저장
            self._save_checkpoint(model, f"lightgbm_weighted_fold{fold+1}.txt")
        
        # 가중치 계산
        self.fold_weights = self._calculate_weights(fold_maes, method=weight_method)
        
        logger.info(f"\n📊 Fold 성능 및 가중치:")
        for i, (mae, weight) in enumerate(zip(fold_maes, self.fold_weights)):
            logger.info(f"  Fold {i+1}: MAE={mae:.4f}, Weight={weight:.4f} ({weight*100:.1f}%)")
        
        # 전체 OOF 성능
        oof_all = np.zeros(len(bundle.x_train))
        for fold, (_, val_idx) in enumerate(gkf.split(bundle.x_train, bundle.y_train, groups=train_groups)):
            X_fold_val = bundle.x_train[val_idx]
            oof_all[val_idx] = self.fold_models[fold].predict(X_fold_val).clip(min=0)
        
        overall_mae = mean_absolute_error(bundle.y_train, oof_all)
        logger.info(f"\n🎯 전체 OOF MAE: {overall_mae:.4f}")
        
        # Validation 성능
        val_pred = self.predict(bundle.x_val)
        val_mae = mean_absolute_error(bundle.y_val, val_pred)
        logger.info(f"🎯 Validation MAE (가중 앙상블): {val_mae:.4f}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """가중치 기반 예측"""
        if not self.fold_models or len(self.fold_weights) == 0:
            raise ValueError("모델이 학습되지 않았습니다. fit_weighted_boosting()을 먼저 실행하세요.")
        
        # 각 모델의 예측을 가중 평균
        predictions = np.zeros(len(X_test))
        for model, weight in zip(self.fold_models, self.fold_weights):
            predictions += weight * model.predict(X_test)
        
        return predictions.clip(min=0)
    
    def _save_checkpoint(self, model: LGBMRegressor, filename: str) -> None:
        """체크포인트 저장"""
        booster = getattr(model, "booster_", None)
        if booster is None:
            logger.warning("LightGBM booster가 없어 체크포인트를 건너뜁니다.")
            return
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(checkpoint_path))


def _apply_physical_constraints(predictions: np.ndarray, test_meta: pl.DataFrame) -> np.ndarray:
    """물리적 제약 조건 적용"""
    df = test_meta.with_columns(
        pl.Series("nins", np.asarray(predictions, dtype=np.float32))
    )

    # KST 시간대 정규화
    df = ensure_kst_time(df, "time")
    
    df = df.with_columns([
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.month().alias("month"),
    ])

    # 야간 일사량 0
    df = df.with_columns([
        pl.when(pl.col("hour").is_between(5, 20, closed="both"))
        .then(pl.col("nins"))
        .otherwise(0.0)
        .alias("nins")
    ])

    # 계절별 상한선
    df = df.with_columns([
        pl.when(pl.col("month").is_in([12, 1, 2]))
        .then(pl.col("nins").clip(0, 900))
        .when(pl.col("month").is_in([6, 7, 8]))
        .then(pl.col("nins").clip(0, 1200))
        .otherwise(pl.col("nins").clip(0, 1100))
        .alias("nins")
    ])

    return df["nins"].to_numpy().astype(np.float32, copy=False)


def run_weighted_boosting_pipeline(
    config: TrainingConfig,
    weight_method: str = "inverse",
    n_splits: int = 5,
):
    """
    가중치 기반 LightGBM 앙상블 파이프라인
    
    Args:
        config: 학습 설정
        weight_method: 가중치 계산 방법
            - "inverse": 1/MAE (추천)
            - "softmax": 성능 차이 증폭
            - "rank": 순위 기반
        n_splits: Fold 개수
    """
    config.resolve_paths()
    set_seed(config.seed)
    
    # 데이터 준비
    from .lgbm import prepare_data_no_scaling
    bundle = prepare_data_no_scaling(config)
    
    logger.info(f"LightGBM Weighted Boosting 학습 시작 (method={weight_method})")
    
    # 가중치 앙상블 학습
    model = LightGBMWeightedBoosting(config)
    model.fit_weighted_boosting(bundle, weight_method=weight_method, n_splits=n_splits)
    
    # 예측
    logger.info("🔮 Test 데이터 예측 중...")
    predictions = model.predict(bundle.x_test)
    
    logger.info(f"📊 예측 통계 (제약 전):")
    logger.info(f"  - 평균: {predictions.mean():.2f}")
    logger.info(f"  - 최대: {predictions.max():.2f}")
    logger.info(f"  - 최소: {predictions.min():.2f}")
    
    # 물리적 제약 후처리
    predictions = _apply_physical_constraints(predictions, bundle.test_meta)
    
    # 제출 파일 저장
    from .pipeline import _save_submission
    submission_path = _save_submission(predictions, bundle, config)
    
    logger.info(f"✅ 제출 파일 저장 완료: {submission_path}")
    
    return {
        'method': f'lightgbm_weighted_boosting_{weight_method}',
        'submission_path': str(submission_path),
        'weights': model.fold_weights.tolist(),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="LightGBM 가중치 앙상블 학습")
    parser.add_argument('--data-dir', type=Path, default=Path('./data'))
    parser.add_argument('--submission-name', type=str, default='lightgbm_weighted_submission.csv')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--weight-method', type=str, default='inverse', 
                        choices=['inverse', 'softmax', 'rank'],
                        help='가중치 계산 방법')
    parser.add_argument('--n-splits', type=int, default=5, help='Fold 개수')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        submission_name=args.submission_name,
        use_gpu=not args.cpu,
        seed=args.seed,
    )
    
    result = run_weighted_boosting_pipeline(
        config,
        weight_method=args.weight_method,
        n_splits=args.n_splits,
    )
    print(f"\n🎉 완료!")
    print(f"  방법: {result['method']}")
    print(f"  가중치: {result['weights']}")
    print(f"  결과: {result['submission_path']}")