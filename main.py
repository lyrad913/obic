from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.config import TrainingConfig
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OBIC irradiance estimator")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to directory containing train/test/submission CSV files")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory to store checkpoints and submissions")
    parser.add_argument("--submission-name", type=str, default=None, help="Filename for generated submission CSV")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum training epochs")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation split ratio based on pv_id groups")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision even if CUDA is available")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.artifacts_dir is not None:
        config.artifacts_dir = args.artifacts_dir
    if args.submission_name is not None:
        config.submission_name = args.submission_name
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        config.eval_batch_size = args.eval_batch_size
    if args.learning_rate is not None:
        config.lr = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.patience is not None:
        config.patience = args.patience
    if args.seed is not None:
        config.seed = args.seed
    if args.no_amp:
        config.amp = False


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    apply_overrides(config, args)
    logger.info(f"Starting pipeline with config: {config}")
    run_pipeline(config)


if __name__ == "__main__":
    main()
