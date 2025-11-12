"""
Pro Stacking 앙상블 - MAE 40 이하 목표
참고: Kaggle 우승 전략 (6개 Base Models + Elastic-Net)

주요 개선:
1. Base Models 6개로 확장 (다양성 극대화)
   - LightGBM (2종)
   - XGBoost
   - HistGradientBoosting (sklearn)
   - Neural Network
   - ExtraTrees (랜덤성 강화)
2. Meta-Model을 Elastic-Net으로 교체 (L1+L2 정규화)
3. 2-Level Stacking 고려
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from .config import TrainingConfig
from .data import DataBundle, prepare_data
from .lgbm import prepare_data_no_scaling
from .model import FeedForwardRegressor
from .utils import resolve_device


class ProStackingEnsemble:
    """
    Pro-level Stacking 앙상블
    
    전략:
    - 6개 Base Models (극대화된 다양성)
    - Elastic-Net Meta-Model (L1+L2 정규화)
    - GroupKFold OOF (정확한 일반화)
    """

    CHECKPOINT_SUFFIXES: dict[str, str] = {
        "neural_net": ".pth",
        "xgboost": ".txt",
        "histgb": ".pkl",
        "extratrees": ".pkl",
    }
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = resolve_device()
        self.base_models = self._init_base_models()
        
        # Elastic-Net: Ridge(L2) + Lasso(L1) 결합
        # alpha: 전체 정규화 강도, l1_ratio: L1 비율
        self.meta_model = ElasticNet(
            alpha=1.0,           # 정규화 강도
            l1_ratio=0.3,        # L1 30%, L2 70%
            random_state=config.seed,
            max_iter=5000,
        )
        
        self.checkpoint_dir = config.checkpoint_dir
        self.scaled_bundle = None

    def _checkpoint_filename(self, model_name: str, fold: int) -> str:
        suffix = self.CHECKPOINT_SUFFIXES.get(model_name, ".txt")
        return f"{model_name}_fold{fold}{suffix}"

    def _checkpoint_path(self, model_name: str, fold: int) -> Path:
        return self.checkpoint_dir / self._checkpoint_filename(model_name, fold)
    
    def _init_base_models(self) -> dict[str, Any]:
        """6개 Base Models 초기화 - 극대화된 다양성"""
        use_gpu = self.config.use_gpu
        seed = self.config.seed
        
        logger.info("📦 Pro Base Models 초기화 중...")
        logger.info("  [전략] 6개 모델로 다양성 극대화")
        
        models = {
            # === 트리 기반 (4개) ===
            
            # 1. LightGBM - Balanced
            'lgbm_balanced': LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.03,
                max_depth=8,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=seed,
                device="gpu" if use_gpu else "cpu",
                verbose=-1,
                objective="mae",
            ),
            
            # 2. LightGBM - Conservative (안정성)
            'lgbm_conservative': LGBMRegressor(
                n_estimators=2000,
                learning_rate=0.02,
                max_depth=5,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
                reg_alpha=0.5,
                reg_lambda=0.5,
                random_state=seed + 1,
                device="gpu" if use_gpu else "cpu",
                verbose=-1,
                objective="mae",
            ),
            
            # 3. XGBoost
            'xgboost': XGBRegressor(
                n_estimators=1200,
                learning_rate=0.04,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.2,
                reg_lambda=0.2,
                random_state=seed,
                tree_method="gpu_hist" if use_gpu else "hist",
                objective="reg:absoluteerror",
                eval_metric="mae",
                verbosity=0,
            ),
            
            # 4. HistGradientBoosting (sklearn - 빠르고 강력)
            'histgb': HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.05,
                max_depth=8,
                l2_regularization=0.1,
                random_state=seed,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=0,
            ),
            
            # === 랜덤성 강화 (1개) ===
            
            # 5. ExtraTrees (극도로 랜덤화된 트리)
            'extratrees': ExtraTreesRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=seed,
                n_jobs=-1,
                verbose=0,
            ),
            
            # === 신경망 (1개) ===
            
            # 6. Neural Network
            'neural_net': 'placeholder',
        }
        
        logger.info(f"✅ {len(models)}개 Base Models 준비 완료")
        logger.info("  - LightGBM: 2종 (균형형 + 보수형)")
        logger.info("  - XGBoost: 1종 (다른 구현)")
        logger.info("  - HistGradientBoosting: 1종 (sklearn)")
        logger.info("  - ExtraTrees: 1종 (랜덤성)")
        logger.info("  - Neural Network: 1종 (비선형)")
        
        return models
    
    def _load_existing_checkpoint(
        self,
        model_name: str,
        fold: int,
        template_model: Any,
        feature_count: int,
    ) -> Any | None:
        """기존 체크포인트 로드"""
        checkpoint_path = self._checkpoint_path(model_name, fold)
        
        if not checkpoint_path.exists():
            return None
            
        try:
            if isinstance(template_model, LGBMRegressor):
                booster = lgb.Booster(model_file=str(checkpoint_path))
                model = LGBMRegressor()
                model._Booster = booster
                model.fitted_ = True
                model._n_features_in_ = feature_count
                model.n_features_in_ = feature_count
                model._n_features = feature_count
            elif isinstance(template_model, XGBRegressor):
                model = deepcopy(template_model)
                model.load_model(str(checkpoint_path))
            elif isinstance(template_model, (HistGradientBoostingRegressor, ExtraTreesRegressor)):
                model = joblib.load(checkpoint_path)
            else:
                logger.debug(f"    ⚠️ 지원되지 않는 체크포인트 타입: {model_name}")
                return None
            
            logger.info(f"    ✅ 체크포인트 로드: {model_name} fold {fold}")
            return model
        except Exception as e:
            logger.warning(f"    체크포인트 로드 실패 ({model_name} fold {fold}): {e}")
            return None
    
    def _save_checkpoint(self, model: Any, filename: str) -> None:
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), str(checkpoint_path))
        elif isinstance(model, (HistGradientBoostingRegressor, ExtraTreesRegressor)):
            joblib.dump(model, checkpoint_path)
        elif hasattr(model, 'booster_'):
            model.booster_.save_model(str(checkpoint_path))
        elif hasattr(model, 'save_model'):
            model.save_model(str(checkpoint_path))
        
        logger.debug(f"💾 저장: {checkpoint_path}")
    
    def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_count: int,
        fold: int,
    ) -> nn.Module:
        """신경망 모델 학습"""
        
        checkpoint_path = self._checkpoint_path("neural_net", fold)
        if checkpoint_path.exists():
            try:
                model = FeedForwardRegressor(
                    input_dim=feature_count,
                    hidden_dims=self.config.hidden_dims,
                    dropout=self.config.dropout,
                ).to(self.device)
                model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                logger.info(f"    ✅ 신경망 체크포인트 로드: fold {fold}")
                return model
            except Exception as e:
                logger.warning(f"    신경망 체크포인트 로드 실패: {e}")
        
        logger.debug(f"    🧠 신경망 학습 시작 (fold {fold})")
        
        model = FeedForwardRegressor(
            input_dim=feature_count,
            hidden_dims=self.config.hidden_dims,
            dropout=self.config.dropout,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.SmoothL1Loss()
        
        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float()
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        max_epochs = 80
        
        for epoch in range(max_epochs):
            model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self._save_checkpoint(model, self._checkpoint_filename("neural_net", fold))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return model
    
    def _predict_neural_network(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """신경망 예측"""
        model.eval()
        dataset = TensorDataset(torch.from_numpy(X).float())
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                preds.append(outputs.cpu().numpy())
        
        return np.concatenate(preds).clip(min=0)
    
    def fit(self, bundle: DataBundle, use_existing_checkpoints: bool = True) -> None:
        """Pro Stacking 학습"""
        logger.info("=" * 80)
        logger.info("🚀 [Pro Stacking Ensemble] 학습 시작")
        logger.info("=" * 80)
        logger.info(f"전략: 6개 Base Models + Elastic-Net Meta-Model")
        logger.info("=" * 80)
        
        # 신경망용 데이터
        logger.info("\n🔧 신경망용 스케일링 데이터 준비...")
        self.scaled_bundle = prepare_data(self.config)
        
        n_splits = 5
        gkf = GroupKFold(n_splits=n_splits)
        train_groups = bundle.train_meta.select(self.config.group_col).to_numpy().ravel()
        
        # Stage 1: Base Models OOF
        logger.info(f"\n📊 [Stage 1] Base Models OOF 학습 ({n_splits}-Fold)")
        oof_predictions = {}
        test_predictions = {}
        val_predictions = {}
        
        for model_name, model in self.base_models.items():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📈 [{model_name.upper()}]")
            logger.info(f"{'=' * 60}")
            
            # 신경망
            if model_name == 'neural_net':
                oof_train = np.zeros(len(self.scaled_bundle.x_train))
                test_preds_folds = []
                val_preds_folds = []
                
                for fold, (train_idx, val_idx) in enumerate(
                    gkf.split(self.scaled_bundle.x_train, self.scaled_bundle.y_train, groups=train_groups)
                ):
                    logger.info(f"  📁 Fold {fold+1}/{n_splits}")
                    
                    nn_model = self._train_neural_network(
                        self.scaled_bundle.x_train[train_idx],
                        self.scaled_bundle.y_train[train_idx],
                        self.scaled_bundle.x_train[val_idx],
                        self.scaled_bundle.y_train[val_idx],
                        len(self.scaled_bundle.feature_names),
                        fold + 1
                    )
                    
                    oof_train[val_idx] = self._predict_neural_network(
                        nn_model, self.scaled_bundle.x_train[val_idx]
                    )
                    test_preds_folds.append(
                        self._predict_neural_network(nn_model, self.scaled_bundle.x_test)
                    )
                    if (
                        self.scaled_bundle.x_val is not None
                        and len(self.scaled_bundle.x_val) > 0
                    ):
                        val_preds_folds.append(
                            self._predict_neural_network(nn_model, self.scaled_bundle.x_val)
                        )
                    
                    fold_mae = mean_absolute_error(
                        self.scaled_bundle.y_train[val_idx], oof_train[val_idx]
                    )
                    logger.info(f"    ✓ Fold {fold+1} MAE: {fold_mae:.4f}")
                
                oof_mae = mean_absolute_error(self.scaled_bundle.y_train, oof_train)
                logger.info(f"\n  🎯 Overall OOF MAE: {oof_mae:.4f}")
                
                oof_predictions[model_name] = oof_train
                test_predictions[model_name] = np.mean(test_preds_folds, axis=0)
                if val_preds_folds:
                    val_predictions[model_name] = np.mean(val_preds_folds, axis=0)
                continue
            
            # 트리 모델
            oof_train = np.zeros(len(bundle.x_train))
            test_preds_folds = []
            val_preds_folds = []
            
            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(bundle.x_train, bundle.y_train, groups=train_groups)
            ):
                logger.info(f"  📁 Fold {fold+1}/{n_splits}")
                
                X_fold_train = bundle.x_train[train_idx]
                y_fold_train = bundle.y_train[train_idx]
                X_fold_val = bundle.x_train[val_idx]
                y_fold_val = bundle.y_train[val_idx]
                
                # 체크포인트 로드
                loaded_model = None
                if use_existing_checkpoints:
                    loaded_model = self._load_existing_checkpoint(
                        model_name, fold + 1, model, len(bundle.feature_names)
                    )
                
                if loaded_model is not None:
                    fold_model = loaded_model
                else:
                    fold_model = deepcopy(model)
                    
                    # sklearn 모델
                    if isinstance(fold_model, (HistGradientBoostingRegressor, ExtraTreesRegressor)):
                        fold_model.fit(X_fold_train, y_fold_train)
                    # LightGBM, XGBoost
                    else:
                        fold_model.fit(
                            X_fold_train, y_fold_train,
                            eval_set=[(X_fold_val, y_fold_val)],
                        )
                    
                    self._save_checkpoint(
                        fold_model,
                        self._checkpoint_filename(model_name, fold + 1)
                    )
                
                oof_train[val_idx] = fold_model.predict(X_fold_val).clip(min=0)
                test_preds_folds.append(fold_model.predict(bundle.x_test).clip(min=0))
                if bundle.x_val is not None and len(bundle.x_val) > 0:
                    val_preds_folds.append(fold_model.predict(bundle.x_val).clip(min=0))
                
                fold_mae = mean_absolute_error(y_fold_val, oof_train[val_idx])
                logger.info(f"    ✓ Fold {fold+1} MAE: {fold_mae:.4f}")
            
            oof_mae = mean_absolute_error(bundle.y_train, oof_train)
            logger.info(f"\n  🎯 Overall OOF MAE: {oof_mae:.4f}")
            
            oof_predictions[model_name] = oof_train
            test_predictions[model_name] = np.mean(test_preds_folds, axis=0)
            if val_preds_folds:
                val_predictions[model_name] = np.mean(val_preds_folds, axis=0)
        
        # Stage 2: Elastic-Net Meta-Model
        logger.info("\n" + "=" * 80)
        logger.info("🎯 [Stage 2] Elastic-Net Meta-Model 학습")
        logger.info("=" * 80)
        
        X_meta = np.column_stack([oof_predictions[name] for name in self.base_models.keys()])
        
        logger.info(f"  Meta 입력 shape: {X_meta.shape}")
        logger.info(f"  각 Base Model 기여도:")
        for i, name in enumerate(self.base_models.keys()):
            mae_single = mean_absolute_error(bundle.y_train, X_meta[:, i])
            logger.info(
                f"    [{name:20s}] "
                f"단독 MAE={mae_single:5.2f}, "
                f"평균={X_meta[:, i].mean():6.2f}, "
                f"std={X_meta[:, i].std():5.2f}"
            )
        
        self.meta_model.fit(X_meta, bundle.y_train)
        
        # Elastic-Net 가중치
        coefficients = self.meta_model.coef_
        logger.info(f"\n  Elastic-Net 가중치:")
        for i, name in enumerate(self.base_models.keys()):
            logger.info(f"    [{name:20s}] weight={coefficients[i]:+.4f}")
        logger.info(f"  Intercept: {self.meta_model.intercept_:.4f}")
        
        # 최종 OOF
        final_oof = self.meta_model.predict(X_meta).clip(min=0)
        final_oof_mae = mean_absolute_error(bundle.y_train, final_oof)
        
        logger.info(f"\n✨ 최종 Train OOF MAE: {final_oof_mae:.4f}")
        
        # Validation
        if bundle.x_val is None or len(bundle.x_val) == 0:
            logger.info("\n📊 Validation 세트가 없어 검증을 스킵합니다.")
        else:
            logger.info(f"\n📊 Validation 평가...")
            val_base_preds = []
            
            for model_name in self.base_models.keys():
                val_pred = val_predictions.get(model_name)
                if val_pred is None or len(val_pred) == 0:
                    logger.warning(f"  [{model_name:20s}] Validation 예측이 없어 스킵합니다.")
                    continue
                
                val_base_preds.append(val_pred)
                val_mae = mean_absolute_error(bundle.y_val, val_pred)
                logger.info(f"  [{model_name:20s}] Val MAE: {val_mae:.4f}")
            
            if not val_base_preds:
                logger.warning("Validation 예측이 없어 최종 MAE를 계산할 수 없습니다.")
            else:
                X_meta_val = np.column_stack(val_base_preds)
                val_pred_final = self.meta_model.predict(X_meta_val).clip(min=0)
                val_mae = mean_absolute_error(bundle.y_val, val_pred_final)
                
                logger.info(f"\n🎯 최종 Validation MAE: {val_mae:.4f}")
                
                if val_mae < 40:
                    logger.info("🎉 목표 달성! MAE 40 이하")
                elif val_mae < 45:
                    logger.info("✅ 우수! Top 7 진입 가능")
                else:
                    logger.info("⚠️  추가 튜닝 필요")
        
        logger.info("=" * 80)
        
        # Test 예측
        self.test_meta_features = np.column_stack([
            test_predictions[name] for name in self.base_models.keys()
        ])
    
    def predict(self, X_test: np.ndarray = None) -> np.ndarray:
        """Test 예측"""
        final_pred = self.meta_model.predict(self.test_meta_features).clip(min=0)
        return final_pred


def apply_physical_constraints(
    predictions: np.ndarray, 
    test_meta: pl.DataFrame
) -> np.ndarray:
    """물리적 제약 (완화)"""
    logger.info("\n🔧 물리적 제약 적용...")
    
    df = test_meta.with_columns(pl.Series("nins", predictions.astype(np.float32)))
    
    df = df.with_columns([
        pl.col("time").dt.hour().alias("hour"),
        pl.col("time").dt.month().alias("month"),
    ])
    
    # 야간 제로
    df = df.with_columns([
        pl.when(pl.col("hour").is_between(5, 20, closed="both"))
        .then(pl.col("nins"))
        .otherwise(0.0)
        .alias("nins")
    ])
    
    # 계절별 상한
    df = df.with_columns([
        pl.when(pl.col("month").is_in([12, 1, 2]))
        .then(pl.col("nins").clip(0, 900))
        .when(pl.col("month").is_in([6, 7, 8]))
        .then(pl.col("nins").clip(0, 1200))
        .otherwise(pl.col("nins").clip(0, 1100))
        .alias("nins")
    ])
    
    result = df["nins"].to_numpy().astype(np.float32)
    
    logger.info(f"  ✓ 평균: {result.mean():.2f} W/m²")
    logger.info(f"  ✓ 최대: {result.max():.2f} W/m²")
    logger.info(f"  ✓ 제로 비율: {(result == 0).sum() / len(result) * 100:.1f}%")
    
    return result


def run_pro_stacking_pipeline(
    config: TrainingConfig,
    use_existing_checkpoints: bool = True,
) -> dict[str, Any]:
    """Pro Stacking 파이프라인"""
    config.resolve_paths()
    
    from .utils import set_seed
    set_seed(config.seed)
    
    bundle = prepare_data_no_scaling(config)
    
    logger.info("\n" + "=" * 80)
    logger.info("🌟 Pro Stacking Ensemble Pipeline")
    logger.info("=" * 80)
    logger.info(f"목표: MAE 40 이하")
    logger.info(f"  - 6개 Base Models (극대화된 다양성)")
    logger.info(f"  - Elastic-Net Meta-Model (L1+L2)")
    logger.info("=" * 80)
    
    model = ProStackingEnsemble(config)
    model.fit(bundle, use_existing_checkpoints=use_existing_checkpoints)
    
    logger.info("\n🔮 Test 예측...")
    predictions = model.predict()
    
    logger.info(f"\n📊 예측 통계 (제약 전):")
    logger.info(f"  - 평균: {predictions.mean():.2f}")
    logger.info(f"  - 최대: {predictions.max():.2f}")
    logger.info(f"  - 최소: {predictions.min():.2f}")
    
    predictions = apply_physical_constraints(predictions, bundle.test_meta)
    
    from .pipeline import _save_submission
    submission_path = _save_submission(predictions, bundle, config)
    
    logger.info(f"\n✅ 제출 파일: {submission_path}")
    
    return {
        'method': 'pro_stacking',
        'n_base_models': len(model.base_models),
        'submission_path': str(submission_path),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pro Stacking")
    parser.add_argument('--data-dir', type=Path, default=Path('./data'))
    parser.add_argument('--submission-name', type=str, default='pro_stacking.csv')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-checkpoints', action='store_true')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        submission_name=args.submission_name,
        use_gpu=not args.cpu,
        seed=args.seed,
    )
    
    result = run_pro_stacking_pipeline(
        config,
        use_existing_checkpoints=not args.no_checkpoints,
    )
    
    print(f"\n🎉 완료!")
    print(f"Base Models: {result['n_base_models']}개")
    print(f"제출 파일: {result['submission_path']}")
