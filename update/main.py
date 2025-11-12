"""Entry point for the enhanced OBIC training pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

if __package__ is None or __package__ == "":
    # Allow running via `python update/main.py`
    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PACKAGE_ROOT.parent))
    from update.config import TrainingConfig  # type: ignore
    from update.pipeline import run_pipeline  # type: ignore
    from update.triple_boosting import run_triple_boosting_pipeline  # type: ignore
else:  # pragma: no cover - standard package imports
    from .config import TrainingConfig
    from .pipeline import run_pipeline
    from .triple_boosting import run_triple_boosting_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBIC 2025 Solar Prediction")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with train/test/submission CSV files")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Directory to store checkpoints and submissions")
    parser.add_argument("--submission-name", type=str, default=None, help="Filename for the generated submission CSV")
    parser.add_argument("--model-type", choices=["nn", "triple_boosting"], default="triple_boosting", help="Select the modeling pipeline")
    parser.add_argument("--ensemble-method", choices=["simple", "stacking", "residual_chain"], default="stacking", help="Ensemble strategy for triple boosting")
    parser.add_argument("--max-epochs", type=int, default=None, help="Maximum training epochs for the neural network pipeline")
    parser.add_argument("--val-ratio", type=float, default=None, help="Validation ratio per pv_id group")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size for the neural net")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision even if supported")
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU-aware settings for triple boosting models")
    return parser.parse_args(argv)


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
    if args.use_gpu:
        config.use_gpu = True


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = TrainingConfig()
    apply_overrides(config, args)
    logger.info(f"Starting pipeline with model_type={args.model_type}, ensemble={args.ensemble_method}")

    if args.model_type == "nn":
        logger.info("실행 모드: 신경망 단독")
        result = run_pipeline(config)
    else:
        logger.info(f"실행 모드: Triple Boosting ({args.ensemble_method})")
        result = run_triple_boosting_pipeline(config, args.ensemble_method)

    logger.info(f"완료! 결과: {result}")


if __name__ == "__main__":
    main()