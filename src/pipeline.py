from __future__ import annotations

import importlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, RunningAverage
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import TrainingConfig
from .data import DataBundle, prepare_data
from .metrics import NormalizedMAE
from .model import FeedForwardRegressor, TwoHeadLoss, TwoHeadRegressor
from .utils import resolve_device, set_seed


def _prepare_loaders(bundle: DataBundle, config: TrainingConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    if config.is_two_head:
        if bundle.train_pv_indices is None or bundle.val_pv_indices is None:
            raise ValueError("Two-head model requires pv_id indices in the data bundle.")
        train_dataset = TensorDataset(
            torch.from_numpy(bundle.x_train),
            torch.from_numpy(bundle.train_pv_indices).long(),
            torch.from_numpy(bundle.y_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(bundle.x_val),
            torch.from_numpy(bundle.val_pv_indices).long(),
            torch.from_numpy(bundle.y_val),
        )
    else:
        train_dataset = TensorDataset(
            torch.from_numpy(bundle.x_train),
            torch.from_numpy(bundle.y_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(bundle.x_val),
            torch.from_numpy(bundle.y_val),
        )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def _resolve_amp_mode(device: torch.device, use_amp: bool) -> str | None:
    if not use_amp:
        return None
    if device.type == "cuda":
        return "amp"
    return None


def _log_epoch_metrics(epoch: int, metrics: dict[str, float]) -> None:
    mae = metrics.get("mae", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    nmape = metrics.get("norm_mae", float("nan"))
    logger.info(f"epoch={epoch:03d} val_mae={mae:.4f} val_rmse={rmse:.4f} val_nmape={nmape:.2f}%")


def _predict(
    model: nn.Module,
    features: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
    pv_indices: np.ndarray | None = None,
) -> np.ndarray:
    pin_memory = device.type == "cuda"
    if config.is_two_head:
        if pv_indices is None:
            raise ValueError("Two-head model requires pv_id indices for prediction.")
        dataset = TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(pv_indices).long(),
        )
    else:
        dataset = TensorDataset(torch.from_numpy(features))
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    preds: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if config.is_two_head:
                feature_batch, pv_batch = batch
                feature_batch = feature_batch.to(device)
                pv_batch = pv_batch.to(device)
                logits, regression = model(feature_batch, pv_batch)
                positive_prob = torch.sigmoid(logits)
                positive_energy = torch.relu(regression)
                outputs = (positive_prob * positive_energy).clamp_min(0.0)
            else:
                (feature_batch,) = batch
                feature_batch = feature_batch.to(device)
                outputs = model(feature_batch).clamp_min(0.0)
            preds.append(outputs.to("cpu"))
    if not preds:
        return np.array([], dtype=np.float32)
    return torch.cat(preds).numpy().astype(np.float32, copy=False)


class CheckpointManager:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.max_checkpoints = max(1, config.max_checkpoints)
        self._records: list[tuple[float, Path]] = []
        self.best_path: Path | None = None

    def _build_path(self, epoch: int, mae: float, extension: str) -> Path:
        run_name = self.config.checkpoint_prefix
        filename = f"{run_name}-epoch{epoch:03d}-mae={mae:.4f}{extension}"
        return self.config.checkpoint_dir / filename

    def _register(self, mae: float, path: Path) -> None:
        self._records.append((mae, path))
        self._records.sort(key=lambda item: item[0])
        while len(self._records) > self.max_checkpoints:
            _, worst_path = self._records.pop(-1)
            try:
                worst_path.unlink()
            except FileNotFoundError:
                pass
        self.best_path = self._records[0][1] if self._records else None

    def save_torch_model(self, model: nn.Module, metrics: dict[str, float], epoch: int) -> None:
        mae = metrics.get("mae")
        if mae is None or not math.isfinite(mae):
            return
        path = self._build_path(epoch, mae, ".pt")
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(payload, path)
        self._register(mae, path)

    def save_joblib(self, payload: Any, mae: float, epoch: int) -> Path | None:
        if not math.isfinite(mae):
            return None
        path = self._build_path(epoch, mae, ".joblib")
        joblib.dump(payload, path)
        self._register(mae, path)
        return path


def _save_submission(
    predictions: np.ndarray,
    bundle: DataBundle,
    config: TrainingConfig,
) -> Path:
    if predictions.shape[0] != bundle.test_meta.height:
        raise ValueError("Prediction count does not match test metadata height.")
    pred_df = bundle.test_meta.with_columns(pl.Series("nins", predictions))
    join_keys = ["time", "pv_id", "type"]
    submission = bundle.submission_template.select(join_keys).join(
        pred_df,
        on=join_keys,
        how="left",
    )
    submission = submission.with_columns(pl.col("nins").fill_null(0.0))
    output_path = config.artifacts_dir / config.submission_name
    submission.write_csv(output_path)
    return output_path


def _attach_callbacks(
    trainer,
    evaluator,
    val_loader,
    config: TrainingConfig,
    model: nn.Module,
) -> CheckpointManager:
    checkpoint_manager = CheckpointManager(config)
    RunningAverage(alpha=0.98, output_transform=lambda output: output).attach(trainer, "loss")
    ProgressBar(desc="Training").attach(trainer, metric_names=["loss"])

    def score_function(engine):
        return -engine.state.metrics["mae"]

    handler = EarlyStopping(patience=config.patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_validation(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        _log_epoch_metrics(engine.state.epoch, metrics)
        checkpoint_manager.save_torch_model(model, metrics, engine.state.epoch)
    return checkpoint_manager


def _compute_normalized_mae(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    denominator = np.abs(y_true).mean() + eps
    if denominator <= eps:
        return 0.0
    return float(np.abs(y_true - y_pred).mean() / denominator * 100.0)


def _align_category_levels(df: pd.DataFrame | None, category_levels: dict[str, list[Any]]) -> pd.DataFrame | None:
    if df is None or not category_levels:
        return df
    for column, levels in category_levels.items():
        if column in df.columns:
            df[column] = pd.Categorical(df[column], categories=levels)
    return df


def _run_lightgbm_inference_only(bundle: DataBundle, config: TrainingConfig) -> dict[str, Any]:
    if config.checkpoint_path is None:
        raise ValueError("inference-only 모드에서는 checkpoint_path를 지정해야 합니다.")
    checkpoint_path = config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"LightGBM 체크포인트를 불러와 추론을 수행합니다: {checkpoint_path}")
    payload = joblib.load(checkpoint_path)
    classifier = payload.get("classifier")
    if classifier is None:
        raise ValueError("Checkpoint에는 classifier가 포함되어야 합니다.")
    regressor = payload.get("regressor")
    category_levels = payload.get("category_levels") or {}

    val_df = bundle.val_frame.to_pandas() if bundle.val_frame is not None else None
    test_df = bundle.test_frame.to_pandas()
    if val_df is not None:
        val_df = _align_category_levels(val_df, category_levels)
    test_df = _align_category_levels(test_df, category_levels)

    y_val = bundle.y_val
    val_metrics: dict[str, float] | None = None
    if val_df is not None and y_val.size > 0:
        prob_val = classifier.predict_proba(val_df)[:, 1]
        reg_val = np.zeros_like(y_val, dtype=np.float64)
        if regressor is not None:
            reg_val = np.clip(regressor.predict(val_df), 0.0, None)
        val_predictions = np.clip(prob_val * reg_val, 0.0, None)
        val_mae = mean_absolute_error(y_val, val_predictions)
        val_rmse = math.sqrt(mean_squared_error(y_val, val_predictions))
        val_nmape = _compute_normalized_mae(y_val, val_predictions)
        val_metrics = {
            "mae": val_mae,
            "rmse": val_rmse,
            "nmape": val_nmape,
        }
        logger.info(f"Validation metrics (inference-only): {json.dumps(val_metrics, default=float)}")

    prob_test = classifier.predict_proba(test_df)[:, 1]
    reg_test = np.zeros(test_df.shape[0], dtype=np.float64)
    if regressor is not None:
        reg_test = np.clip(regressor.predict(test_df), 0.0, None)
    test_predictions = np.clip(prob_test * reg_test, 0.0, None).astype(np.float32, copy=False)

    submission_path = _save_submission(test_predictions, bundle, config)
    logger.info(f"Submission saved to {submission_path}")

    result: dict[str, Any] = {
        "device": "cpu",
        "feature_count": len(bundle.feature_names),
        "submission_path": str(submission_path),
        "loaded_checkpoint": str(checkpoint_path),
    }
    if val_metrics is not None:
        result["val_metrics"] = val_metrics
    return result


def _predict_lightgbm_heads(
    classifier: Any,
    regressor: Any | None,
    frame: pd.DataFrame,
) -> np.ndarray:
    prob = classifier.predict_proba(frame)[:, 1]
    reg = np.zeros(frame.shape[0], dtype=np.float64)
    if regressor is not None:
        reg = np.clip(regressor.predict(frame), 0.0, None)
    return np.clip(prob * reg, 0.0, None)


def _run_lightgbm_inference_only(bundle: DataBundle, config: TrainingConfig) -> dict[str, Any]:
    checkpoint_paths: list[Path] = []
    if config.checkpoint_paths is not None:
        checkpoint_paths.extend(Path(p) for p in config.checkpoint_paths)
    if config.checkpoint_path is not None:
        checkpoint_paths.append(Path(config.checkpoint_path))
    if not checkpoint_paths:
        raise ValueError("inference-only 모드에서는 최소 하나 이상의 checkpoint를 지정해야 합니다.")

    val_df_base = bundle.val_frame.to_pandas() if bundle.val_frame is not None else None
    test_df_base = bundle.test_frame.to_pandas()
    y_val = bundle.y_val

    test_predictions_list: list[np.ndarray] = []
    val_predictions_list: list[np.ndarray] = []
    per_model_metrics: list[dict[str, Any]] = []

    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        logger.info(f"Checkpoint 로드 및 추론: {path}")
        payload = joblib.load(path)
        classifier = payload.get("classifier")
        if classifier is None:
            raise ValueError(f"Checkpoint {path}에는 classifier가 포함되어야 합니다.")
        regressor = payload.get("regressor")
        category_levels = payload.get("category_levels") or {}

        val_df = _align_category_levels(val_df_base.copy() if val_df_base is not None else None, category_levels)
        test_df = _align_category_levels(test_df_base.copy(), category_levels)

        model_test_pred = _predict_lightgbm_heads(classifier, regressor, test_df)
        test_predictions_list.append(model_test_pred.astype(np.float32, copy=False))

        if val_df is not None and y_val.size > 0:
            model_val_pred = _predict_lightgbm_heads(classifier, regressor, val_df)
            val_predictions_list.append(model_val_pred.astype(np.float64, copy=False))
            val_mae = mean_absolute_error(y_val, model_val_pred)
            val_rmse = math.sqrt(mean_squared_error(y_val, model_val_pred))
            val_nmape = _compute_normalized_mae(y_val, model_val_pred)
            metrics = {
                "mae": val_mae,
                "rmse": val_rmse,
                "nmape": val_nmape,
                "checkpoint": str(path),
            }
            per_model_metrics.append(metrics)
            logger.info(f"Validation metrics (checkpoint={path.name}): {json.dumps(metrics, default=float)}")

    if not test_predictions_list:
        raise RuntimeError("No predictions were generated during inference-only execution.")

    ensemble_test = np.mean(np.stack(test_predictions_list, axis=0), axis=0)

    ensemble_val_metrics: dict[str, float] | None = None
    if val_predictions_list:
        ensemble_val = np.mean(np.stack(val_predictions_list, axis=0), axis=0)
        val_mae = mean_absolute_error(y_val, ensemble_val)
        val_rmse = math.sqrt(mean_squared_error(y_val, ensemble_val))
        val_nmape = _compute_normalized_mae(y_val, ensemble_val)
        ensemble_val_metrics = {
            "mae": val_mae,
            "rmse": val_rmse,
            "nmape": val_nmape,
        }
        logger.info(
            "Validation metrics (ensemble over %d checkpoints): %s",
            len(checkpoint_paths),
            json.dumps(ensemble_val_metrics, default=float),
        )

    submission_path = _save_submission(ensemble_test.astype(np.float32, copy=False), bundle, config)
    logger.info(f"Submission saved to {submission_path}")

    result: dict[str, Any] = {
        "device": "cpu",
        "feature_count": len(bundle.feature_names),
        "submission_path": str(submission_path),
        "loaded_checkpoints": [str(path) for path in checkpoint_paths],
    }
    if ensemble_val_metrics is not None:
        result["val_metrics"] = ensemble_val_metrics
    if per_model_metrics:
        result["per_model_metrics"] = per_model_metrics
    return result


def _run_lightgbm(bundle: DataBundle, config: TrainingConfig) -> dict[str, Any]:
    if config.inference_only:
        return _run_lightgbm_inference_only(bundle, config)

    if bundle.train_frame is None or bundle.val_frame is None or bundle.test_frame is None:
        raise ValueError("LightGBM 실행을 위해서는 train/val/test 프레임이 필요합니다.")

    logger.info("LightGBM 기반 Two-Stage 모델을 학습합니다.")

    try:
        lightgbm = importlib.import_module("lightgbm")
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "LightGBM 패키지가 설치되어 있지 않습니다. `uv pip install lightgbm` 이후 다시 시도해주세요."
        ) from exc

    LGBMClassifier = lightgbm.LGBMClassifier
    LGBMRegressor = lightgbm.LGBMRegressor
    early_stopping = lightgbm.early_stopping
    log_evaluation = lightgbm.log_evaluation

    threshold = config.positive_threshold
    cat_features = list(bundle.categorical_features or [])

    train_pl = bundle.train_frame
    val_pl = bundle.val_frame
    test_pl = bundle.test_frame

    total_train = train_pl.height
    rng = np.random.default_rng(config.seed)
    if 0.0 < config.lgbm_sample_fraction < 1.0:
        subset_size = max(1, int(total_train * config.lgbm_sample_fraction))
        subset_idx = np.sort(rng.choice(total_train, size=subset_size, replace=False))
        logger.info(
            "LightGBM 학습 데이터를 서브샘플링합니다: {}/{} ({:.2f})",
            subset_size,
            total_train,
            config.lgbm_sample_fraction,
        )
        index_frame = pl.DataFrame({"_row_idx": subset_idx.tolist()})
        train_pl_sub = (
            train_pl.with_row_count("_row_idx")
            .join(index_frame, on="_row_idx", how="inner")
            .sort("_row_idx")
            .drop("_row_idx")
        )
        y_train_sub = bundle.y_train[subset_idx]
    else:
        train_pl_sub = train_pl
        y_train_sub = bundle.y_train
    y_val = bundle.y_val

    train_df = train_pl_sub.to_pandas()
    val_df = val_pl.to_pandas()
    test_df = test_pl.to_pandas()

    category_levels: dict[str, list[Any]] = {}
    for cat in cat_features:
        if cat not in train_df.columns:
            continue
        categories = np.unique(
            np.concatenate(
                [
                    train_df[cat].to_numpy(copy=False),
                    val_df[cat].to_numpy(copy=False),
                    test_df[cat].to_numpy(copy=False),
                ]
            )
        )
        train_df[cat] = pd.Categorical(train_df[cat], categories=categories)
        val_df[cat] = pd.Categorical(val_df[cat], categories=categories)
        test_df[cat] = pd.Categorical(test_df[cat], categories=categories)
    category_levels[cat] = categories.tolist()

    y_train_binary = (y_train_sub > threshold).astype(np.int8)
    y_val_binary = (y_val > threshold).astype(np.int8)

    classifier = LGBMClassifier(
        n_estimators=config.lgbm_n_estimators,
        learning_rate=config.lgbm_learning_rate,
        num_leaves=config.lgbm_num_leaves,
        max_depth=config.lgbm_max_depth,
        min_child_samples=config.lgbm_min_child_samples,
        subsample=config.lgbm_bagging_fraction,
        subsample_freq=config.lgbm_bagging_freq,
        colsample_bytree=config.lgbm_feature_fraction,
        reg_lambda=config.lgbm_lambda_l2,
        random_state=config.seed,
        objective="binary",
        n_jobs=config.num_workers if config.num_workers > 0 else -1,
    )

    clf_eval_sets = []
    if val_df.shape[0] > 0:
        clf_eval_sets.append((val_df, y_val_binary))
    clf_callbacks = []
    if config.lgbm_log_evaluation_period > 0:
        clf_callbacks.append(log_evaluation(config.lgbm_log_evaluation_period))
    if clf_eval_sets and config.lgbm_early_stopping_rounds > 0:
        clf_callbacks.append(early_stopping(config.lgbm_early_stopping_rounds, verbose=False))

    classifier.fit(
        train_df,
        y_train_binary,
        eval_set=clf_eval_sets or None,
        eval_metric="binary_logloss",
        categorical_feature=cat_features or "auto",
        callbacks=clf_callbacks or None,
    )
    logger.info("LightGBM 분류 헤드 학습 완료")

    regressor: Any | None = None
    positive_mask = y_train_sub > threshold
    val_positive_mask = y_val > threshold
    if positive_mask.any():
        reg_train_df = train_df.loc[positive_mask]
        reg_target = y_train_sub[positive_mask]
        regressor = LGBMRegressor(
            n_estimators=config.lgbm_n_estimators,
            learning_rate=config.lgbm_learning_rate,
            num_leaves=config.lgbm_num_leaves,
            max_depth=config.lgbm_max_depth,
            min_child_samples=config.lgbm_min_child_samples,
            subsample=config.lgbm_bagging_fraction,
            subsample_freq=config.lgbm_bagging_freq,
            colsample_bytree=config.lgbm_feature_fraction,
            reg_lambda=config.lgbm_lambda_l2,
            random_state=config.seed,
            objective="regression_l1",
            n_jobs=config.num_workers if config.num_workers > 0 else -1,
        )
        reg_eval_sets = []
        if val_positive_mask.any():
            reg_eval_sets.append((val_df.loc[val_positive_mask], y_val[val_positive_mask]))
        reg_callbacks = []
        if config.lgbm_log_evaluation_period > 0:
            reg_callbacks.append(log_evaluation(config.lgbm_log_evaluation_period))
        if reg_eval_sets and config.lgbm_early_stopping_rounds > 0:
            reg_callbacks.append(early_stopping(config.lgbm_early_stopping_rounds, verbose=False))

        regressor.fit(
            reg_train_df,
            reg_target,
            eval_set=reg_eval_sets or None,
            eval_metric="l1",
            categorical_feature=cat_features or "auto",
            callbacks=reg_callbacks or None,
        )
        logger.info("LightGBM 회귀 헤드 학습 완료")
    else:
        logger.warning("양수 타깃 샘플이 없어 회귀 헤드를 학습하지 못했습니다. 모든 양수 추론을 0으로 대체합니다.")

    prob_val = classifier.predict_proba(val_df)[:, 1]
    reg_val = np.zeros_like(y_val, dtype=np.float64)
    if regressor is not None:
        reg_val = np.clip(regressor.predict(val_df), 0.0, None)
    val_predictions = np.clip(prob_val * reg_val, 0.0, None)

    prob_test = classifier.predict_proba(test_df)[:, 1]
    reg_test = np.zeros(test_df.shape[0], dtype=np.float64)
    if regressor is not None:
        reg_test = np.clip(regressor.predict(test_df), 0.0, None)
    test_predictions = np.clip(prob_test * reg_test, 0.0, None).astype(np.float32, copy=False)

    val_mae = mean_absolute_error(y_val, val_predictions)
    val_rmse = math.sqrt(mean_squared_error(y_val, val_predictions))
    val_nmape = _compute_normalized_mae(y_val, val_predictions)

    checkpoint_manager = CheckpointManager(config)
    payload = {
        "classifier": classifier,
        "regressor": regressor,
        "threshold": threshold,
        "categorical_features": cat_features,
        "category_levels": category_levels,
        "metrics": {
            "mae": val_mae,
            "rmse": val_rmse,
            "nmape": val_nmape,
        },
    }
    checkpoint_path = checkpoint_manager.save_joblib(payload, val_mae, epoch=0)
    if checkpoint_path is not None:
        logger.info(f"모델을 저장했습니다: {checkpoint_path}")

    submission_path = _save_submission(test_predictions, bundle, config)

    metrics_payload = {
        "mae": val_mae,
        "rmse": val_rmse,
        "nmape": val_nmape,
    }
    logger.info(f"Validation metrics: {json.dumps(metrics_payload, default=float)}")
    logger.info(f"Submission saved to {submission_path}")

    result: dict[str, Any] = {
        "device": "cpu",
        "feature_count": len(bundle.feature_names),
        "val_metrics": metrics_payload,
        "submission_path": str(submission_path),
    }
    if checkpoint_path is not None:
        result["checkpoint_path"] = str(checkpoint_path)
    return result


def run_pipeline(config: TrainingConfig) -> dict[str, Any]:
    if config.is_tree_model and config.scale_features:
        logger.info("Tree-based model을 선택해 스케일링을 비활성화합니다.")
        config.scale_features = False

    config.resolve_paths()

    if config.inference_only:
        if not config.is_lightgbm:
            raise ValueError("inference-only 모드는 현재 LightGBM 모델에서만 지원됩니다.")
        if config.checkpoint_path is None and not config.checkpoint_paths:
            raise ValueError("inference-only 모드에서는 최소 하나의 checkpoint를 지정해야 합니다.")

    set_seed(config.seed)

    bundle = prepare_data(config)
    logger.info(f"Loaded dataset with {len(bundle.feature_names)} features")
    logger.info(
        f"학습 샘플: {bundle.x_train.shape[0]}, 검증 샘플: {bundle.x_val.shape[0]}, 테스트 샘플: {bundle.x_test.shape[0]}"
    )

    if config.is_tree_model:
        if config.is_hist_gbdt:
            return _run_hist_gradient_boosting(bundle, config)
        if config.is_lightgbm:
            return _run_lightgbm(bundle, config)
        raise ValueError(f"Unsupported tree model name: {config.model_name}")

    device = resolve_device()
    logger.info(f"Using device {device}")

    train_loader, val_loader = _prepare_loaders(bundle, config, device)
    logger.info(
        f"DataLoader 준비 완료 (train_batches={len(train_loader)}, val_batches={len(val_loader)})"
    )

    if config.is_two_head:
        if bundle.pv_id_mapping is None:
            raise ValueError("Two-head model requires pv_id mapping in the data bundle.")
        pv_vocab_size = len(bundle.pv_id_mapping)
        if pv_vocab_size == 0:
            raise ValueError("pv_id vocabulary is empty; cannot build embedding layer.")

        model = TwoHeadRegressor(
            input_dim=len(bundle.feature_names),
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            pv_vocab_size=pv_vocab_size,
            pv_embedding_dim=config.pv_embedding_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        criterion = TwoHeadLoss(
            bce_weight=config.bce_weight,
            regression_weight=config.regression_weight,
            positive_threshold=config.positive_threshold,
            label_smoothing=config.bce_label_smoothing,
        )

        amp_mode = _resolve_amp_mode(device, config.amp)
        amp_enabled = amp_mode == "amp"
        scaler = GradScaler(enabled=amp_enabled)
        autocast_cm = autocast if amp_enabled else nullcontext

        def _two_head_train_step(engine, batch):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            features, pv_idx, targets = batch
            features = features.to(device)
            pv_idx = pv_idx.to(device)
            targets = targets.to(device)
            with autocast_cm():
                logits, regression = model(features, pv_idx)
                loss = criterion((logits, regression), targets)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            return loss.item()

        def _two_head_eval_step(engine, batch):
            model.eval()
            with torch.no_grad():
                features, pv_idx, targets = batch
                features = features.to(device)
                pv_idx = pv_idx.to(device)
                targets = targets.to(device)
                logits, regression = model(features, pv_idx)
                positive_prob = torch.sigmoid(logits)
                positive_energy = torch.relu(regression)
                predictions = (positive_prob * positive_energy).clamp_min(0.0)
            return predictions, targets

        trainer = Engine(_two_head_train_step)
        metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "rmse": RootMeanSquaredError(),
            "norm_mae": NormalizedMAE(),
        }
        evaluator = Engine(_two_head_eval_step)
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
    else:
        model = FeedForwardRegressor(
            input_dim=len(bundle.feature_names),
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        loss_fn = nn.SmoothL1Loss()

        amp_mode = _resolve_amp_mode(device, config.amp)

        trainer = create_supervised_trainer(
            model,
            optimizer,
            loss_fn,
            device=device,
            amp_mode=amp_mode,
        )

        metrics = {
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
            "rmse": RootMeanSquaredError(),
            "norm_mae": NormalizedMAE(),
        }
        evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    checkpoint_manager = _attach_callbacks(
        trainer,
        evaluator,
        val_loader,
        config,
        model,
    )

    logger.info(f"Starting training for up to {config.max_epochs} epochs")
    trainer.run(train_loader, max_epochs=config.max_epochs)

    best_checkpoint_path = checkpoint_manager.best_path
    if best_checkpoint_path is not None:
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        logger.info(f"Loaded best model from {best_checkpoint_path}")
    else:
        logger.warning("No checkpoint was saved during training.")

    val_metrics = evaluator.run(val_loader).metrics
    _log_epoch_metrics(trainer.state.epoch, val_metrics)

    predictions = _predict(
        model,
        bundle.x_test,
        config,
        device,
        bundle.test_pv_indices if config.is_two_head else None,
    )
    submission_path = _save_submission(predictions, bundle, config)

    metrics_payload = {
        "mae": val_metrics.get("mae"),
        "rmse": val_metrics.get("rmse"),
        "nmape": val_metrics.get("norm_mae"),
    }
    logger.info(f"Validation metrics: {json.dumps(metrics_payload, default=float)}")
    logger.info(f"Submission saved to {submission_path}")

    return {
        "device": str(device),
        "feature_count": len(bundle.feature_names),
        "val_metrics": metrics_payload,
        "submission_path": str(submission_path),
    }
