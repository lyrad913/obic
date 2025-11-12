
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

# 상대 import를 절대 import로 변경
# from .config import TrainingConfig  # 원래
# from .data import DataBundle  # 원래

from .config import TrainingConfig
from .data import DataBundle, prepare_data
from .utils import set_seed



class TripleBoostingEnsemble:
    """
    3가지 부스팅 알고리즘 앙상블
    - LightGBM: 빠른 학습, 대용량 데이터에 강함
    - XGBoost: 정확도 높음, GPU 가속 우수
    - CatBoost: 범주형 특성 처리 우수, 과적합 방지
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = self._init_models()
        self.meta_model = Ridge(alpha=1.0)  # Stacking용 메타 모델
        self.feature_names: list[str] | None = None

    def set_feature_names(self, feature_names: Sequence[str]) -> None:
        self.feature_names = list(feature_names)

    def _wrap_features(self, X: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            return X
        if self.feature_names is None:
            raise ValueError("feature_names must be set before wrapping features.")
        return pd.DataFrame(X, columns=self.feature_names)
        
    def _init_models(self) -> dict:
        """3개 모델 초기화"""
        
        use_gpu = self.config.use_gpu

        # LightGBM: 빠른 베이스라인
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
        lgbm = LGBMRegressor(**lgbm_params)
        
        # XGBoost: 높은 정확도
        xgb_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": self.config.seed,
            "tree_method": "gpu_hist" if use_gpu else "hist",
            "objective": "reg:absoluteerror",
            "eval_metric": "mae",
            "early_stopping_rounds": 100,
        }
        if use_gpu:
            xgb_params["predictor"] = "gpu_predictor"
        xgb = XGBRegressor(**xgb_params)
        
        # CatBoost: 강력한 정규화
        catboost = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=7,
            subsample=0.8,
            colsample_bylevel=0.8,
            min_data_in_leaf=20,
            l2_leaf_reg=3.0,
            random_state=self.config.seed,
            task_type="GPU" if use_gpu else "CPU",
            loss_function="MAE",
            eval_metric="MAE",
            verbose=False,
            early_stopping_rounds=100,
        )
        
        return {
            'lightgbm': lgbm,
            'xgboost': xgb,
            'catboost': catboost,
        }
    
    def fit_simple(self, bundle: DataBundle) -> None:
        """
        방법 1: 단순 가중 평균 (빠른 실험용)
        학습 시간: ~20분
        """
        logger.info("🌳 [Simple Weighted Ensemble] 3개 모델 학습 시작")
        train_features = self._wrap_features(bundle.x_train)
        val_features = self._wrap_features(bundle.x_val)
        
        for name, model in self.models.items():
            logger.info(f"  학습 중: {name}")
            
            if name == 'xgboost':
                model.fit(
                    train_features, bundle.y_train,
                    eval_set=[(val_features, bundle.y_val)],
                    verbose=False
                )
            elif name == 'catboost':
                model.fit(
                    train_features, bundle.y_train,
                    eval_set=(val_features, bundle.y_val),
                    verbose=False
                )
            else:  # lightgbm
                model.fit(
                    train_features, bundle.y_train,
                    eval_set=[(val_features, bundle.y_val)],
                )
            
            # 개별 모델 성능
            val_pred = model.predict(val_features).clip(min=0)
            mae = mean_absolute_error(bundle.y_val, val_pred)
            logger.info(f"    {name} 단독 MAE: {mae:.4f}")
        
        # 가중 평균 앙상블 성능
        ensemble_pred = self._weighted_average_predict(val_features)
        ensemble_mae = mean_absolute_error(bundle.y_val, ensemble_pred)
        logger.info(f"🎯 가중 평균 앙상블 MAE: {ensemble_mae:.4f}")
    
    def fit_stacking(self, bundle: DataBundle) -> None:
        """
        방법 2: Stacking (최고 성능)
        학습 시간: ~40분
        """
        logger.info("🏗️ [Stacking Ensemble] 2단계 학습 시작")
        
        # ============================================
        # 1단계: Out-of-Fold 예측 생성
        # ============================================
        logger.info("1단계: OOF(Out-of-Fold) 예측 생성 중...")
        
        oof_train = np.zeros((len(bundle.x_train), len(self.models)))
        oof_val = np.zeros((len(bundle.x_val), len(self.models)))
        full_train_features = self._wrap_features(bundle.x_train)
        full_val_features = self._wrap_features(bundle.x_val)
        
        # 발전소 그룹 기반 5-Fold CV
        gkf = GroupKFold(n_splits=5)
        
        # train 데이터의 pv_id 추출 (bundle.train_meta에서)
        train_groups = bundle.train_meta.select(self.config.group_col).to_numpy().ravel()
        
        for model_idx, (name, model) in enumerate(self.models.items()):
            logger.info(f"  모델 {model_idx+1}/3: {name}")
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(bundle.x_train, bundle.y_train, groups=train_groups)):
                X_fold_train = self._wrap_features(bundle.x_train[train_idx])
                y_fold_train = bundle.y_train[train_idx]
                X_fold_val = self._wrap_features(bundle.x_train[val_idx])
                y_fold_val = bundle.y_train[val_idx]
                
                # Fold 학습 (조기 종료를 위한 검증 세트 지정)
                if name == 'xgboost':
                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        verbose=False,
                    )
                elif name == 'catboost':
                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=(X_fold_val, y_fold_val),
                        verbose=False,
                    )
                else:
                    model.fit(
                        X_fold_train,
                        y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)],
                        # verbose=False,
                    )
                
                # OOF 예측
                oof_train[val_idx, model_idx] = model.predict(X_fold_val).clip(min=0)
            
            # 전체 데이터로 재학습
            logger.info(f"    {name} 전체 데이터로 재학습 중...")
            if name == 'xgboost':
                model.fit(
                    full_train_features, bundle.y_train,
                    eval_set=[(full_val_features, bundle.y_val)],
                    verbose=False
                )
            elif name == 'catboost':
                model.fit(
                    full_train_features, bundle.y_train,
                    eval_set=(full_val_features, bundle.y_val),
                    verbose=False
                )
            else:
                model.fit(
                    full_train_features, bundle.y_train,
                    eval_set=[(full_val_features, bundle.y_val)],
                )
            
            # 검증 세트 예측
            oof_val[:, model_idx] = model.predict(full_val_features).clip(min=0)
            
            # 개별 성능 확인
            fold_mae = mean_absolute_error(bundle.y_train, oof_train[:, model_idx])
            logger.info(f"    {name} OOF MAE: {fold_mae:.4f}")
        
        # ============================================
        # 2단계: 메타 모델 학습
        # ============================================
        logger.info("2단계: 메타 모델(Ridge) 학습 중...")
        self.meta_model.fit(oof_train, bundle.y_train)
        
        # Stacking 성능 평가
        stacking_val_pred = self.meta_model.predict(oof_val).clip(min=0)
        stacking_mae = mean_absolute_error(bundle.y_val, stacking_val_pred)
        
        logger.info(f"🎯 Stacking 앙상블 MAE: {stacking_mae:.4f}")
        logger.info(f"메타 모델 가중치: {self.meta_model.coef_}")
    
    def fit_residual_chain(self, bundle: DataBundle) -> None:
        """
        방법 3: 잔차 체인 (창의성 높음)
        LightGBM → XGBoost(잔차) → CatBoost(잔차)
        학습 시간: ~30분
        """
        logger.info("⛓️ [Residual Chain] 3단계 잔차 학습 시작")
        
        current_train_residual = bundle.y_train.copy()
        current_val_residual = bundle.y_val.copy()
        train_features = self._wrap_features(bundle.x_train)
        val_features = self._wrap_features(bundle.x_val)
        
        train_predictions = np.zeros(len(bundle.y_train))
        val_predictions = np.zeros(len(bundle.y_val))
        
        for i, (name, model) in enumerate(self.models.items()):
            logger.info(f"Stage {i+1}/3: {name} 학습 중...")
            
            # 현재 잔차 학습
            if name == 'xgboost':
                model.fit(
                    train_features, current_train_residual,
                    eval_set=[(val_features, current_val_residual)],
                    verbose=False
                )
            elif name == 'catboost':
                model.fit(
                    train_features, current_train_residual,
                    eval_set=(val_features, current_val_residual),
                    verbose=False
                )
            else:
                model.fit(
                    train_features, current_train_residual,
                    eval_set=[(val_features, current_val_residual)],
                )
            
            # 예측
            train_pred = model.predict(train_features).clip(min=0)
            val_pred = model.predict(val_features).clip(min=0)
            
            # 누적 예측
            train_predictions += train_pred
            val_predictions += val_pred
            
            # 다음 단계를 위한 잔차 업데이트
            current_train_residual = bundle.y_train - train_predictions
            current_val_residual = bundle.y_val - val_predictions
            
            # 현재까지 누적 성능
            stage_mae = mean_absolute_error(bundle.y_val, val_predictions)
            logger.info(f"  Stage {i+1} 누적 MAE: {stage_mae:.4f}")
            logger.info(f"  잔차 평균: {current_val_residual.mean():.4f}, 표준편차: {current_val_residual.std():.4f}")
        
        logger.info(f"🎯 최종 잔차 체인 MAE: {stage_mae:.4f}")
    
    def _weighted_average_predict(self, X: np.ndarray, weights: dict = None) -> np.ndarray:
        """가중 평균 예측"""
        if weights is None:
            # 기본 가중치 (실험적으로 조정 가능)
            weights = {
                'lightgbm': 0.35,
                'xgboost': 0.35,
                'catboost': 0.30,
            }
        
        X_wrapped = self._wrap_features(X)
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_wrapped).clip(min=0)
            predictions.append(pred * weights[name])
        
        return np.sum(predictions, axis=0)
    
    def predict_simple(self, X_test: np.ndarray) -> np.ndarray:
        """단순 가중 평균 예측"""
        return self._weighted_average_predict(X_test)
    
    def predict_stacking(self, X_test: np.ndarray) -> np.ndarray:
        """Stacking 예측"""
        # 각 모델의 예측을 메타 특성으로 사용
        X_wrapped = self._wrap_features(X_test)
        meta_features = np.column_stack([
            model.predict(X_wrapped).clip(min=0) 
            for model in self.models.values()
        ])
        return self.meta_model.predict(meta_features).clip(min=0)
    
    def predict_residual_chain(self, X_test: np.ndarray) -> np.ndarray:
        """잔차 체인 예측"""
        X_wrapped = self._wrap_features(X_test)
        prediction = np.zeros(len(X_test))
        for model in self.models.values():
            pred = model.predict(X_wrapped).clip(min=0)
            prediction += pred
        return prediction


# ========================================
# 사용 예시
# ========================================
def run_triple_boosting_pipeline(config: TrainingConfig, ensemble_method: str = 'stacking'):
    """
    Triple Boosting 파이프라인 실행
    
    Args:
        config: TrainingConfig
        ensemble_method: 'simple', 'stacking', 'residual_chain'
    """
    from .data import prepare_data
    from .utils import set_seed
    
    config.resolve_paths()
    set_seed(config.seed)
    bundle = prepare_data(config)
    
    logger.info(f"Triple Boosting Ensemble 시작 (method={ensemble_method})")
    
    ensemble = TripleBoostingEnsemble(config)
    ensemble.set_feature_names(bundle.feature_names)
    
    # 학습
    if ensemble_method == 'simple':
        ensemble.fit_simple(bundle)
        predictions = ensemble.predict_simple(bundle.x_test)
    elif ensemble_method == 'stacking':
        ensemble.fit_stacking(bundle)
        predictions = ensemble.predict_stacking(bundle.x_test)
    elif ensemble_method == 'residual_chain':
        ensemble.fit_residual_chain(bundle)
        predictions = ensemble.predict_residual_chain(bundle.x_test)
    else:
        raise ValueError(f"Unknown method: {ensemble_method}")
    
    # 물리적 제약 후처리
    predictions = _apply_physical_constraints(predictions, bundle.test_meta)
    
    # 제출 파일 저장
    from .pipeline import _save_submission
    submission_path = _save_submission(predictions, bundle, config)
    
    logger.info(f"✅ 제출 파일 저장 완료: {submission_path}")
    
    return {
        'method': ensemble_method,
        'submission_path': str(submission_path),
    }


def _apply_physical_constraints(predictions: np.ndarray, test_meta) -> np.ndarray:
    """물리적 제약 후처리"""
    import polars as pl
    
    df = test_meta.with_columns(
        pl.Series('nins', np.asarray(predictions, dtype=np.float32))
    )

    if 'time' not in df.columns:
        return np.clip(df['nins'].to_numpy(), a_min=0.0, a_max=None)

    df = df.with_columns([
        pl.col('time').dt.hour().alias('hour'),
        pl.col('time').dt.month().alias('month'),
    ])

    df = df.with_columns([
        pl.when(pl.col('hour').is_between(6, 18, closed='both'))
        .then(pl.col('nins'))
        .otherwise(0.0)
        .alias('nins')
    ])

    df = df.with_columns([
        pl.when(pl.col('month').is_in([12, 1, 2]))
        .then(pl.col('nins').clip(0, 700))
        .when(pl.col('month').is_in([6, 7, 8]))
        .then(pl.col('nins').clip(0, 1000))
        .otherwise(pl.col('nins').clip(0, 850))
        .alias('nins')
    ])
    
    return df['nins'].to_numpy().astype(np.float32, copy=False)