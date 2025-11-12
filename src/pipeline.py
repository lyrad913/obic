from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError, RunningAverage
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

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
    checkpoint_dir: Path,
    model: nn.Module,
) -> ModelCheckpoint:
    RunningAverage(alpha=0.98, output_transform=lambda output: output).attach(trainer, "loss")
    ProgressBar(desc="Training").attach(trainer, metric_names=["loss"])

    def score_function(engine):
        return -engine.state.metrics["mae"]

    handler = EarlyStopping(patience=config.patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, handler)

    checkpointer = ModelCheckpoint(
        dirname=str(checkpoint_dir),
        filename_prefix="regressor",
        n_saved=1,
        score_name="val_mae",
        score_function=score_function,
        global_step_transform=lambda *_: trainer.state.epoch,
        require_empty=False,
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpointer, {"model": model})

    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_validation(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        _log_epoch_metrics(engine.state.epoch, metrics)

    return checkpointer


def run_pipeline(config: TrainingConfig) -> dict[str, Any]:
    config.resolve_paths()
    set_seed(config.seed)
    device = resolve_device()
    logger.info(f"Using device {device}")

    bundle = prepare_data(config)
    logger.info(f"Loaded dataset with {len(bundle.feature_names)} features")
    logger.info(
        f"학습 샘플: {bundle.x_train.shape[0]}, 검증 샘플: {bundle.x_val.shape[0]}, 테스트 샘플: {bundle.x_test.shape[0]}"
    )

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

    checkpointer = _attach_callbacks(
        trainer,
        evaluator,
        val_loader,
        config,
        config.checkpoint_dir,
        model,
    )

    logger.info(f"Starting training for up to {config.max_epochs} epochs")
    trainer.run(train_loader, max_epochs=config.max_epochs)

    if checkpointer.last_checkpoint is not None:
        checkpoint = torch.load(checkpointer.last_checkpoint, map_location=device)
        state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if state_dict is None:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        logger.info(f"Loaded best model from {checkpointer.last_checkpoint}")
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
