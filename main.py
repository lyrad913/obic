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
    parser.add_argument("--model", choices=["feedforward", "two_head", "hist_gbdt", "lightgbm"], default=None, help="Model architecture to train")
    parser.add_argument("--pv-embedding-dim", type=int, default=None, help="Embedding dimension for pv_id when using two_head model")
    parser.add_argument("--bce-weight", type=float, default=None, help="Loss weight for the classification head in the two_head model")
    parser.add_argument("--reg-weight", type=float, default=None, help="Loss weight for the regression head in the two_head model")
    parser.add_argument("--positive-threshold", type=float, default=None, help="Threshold to decide positive samples for the regression head")
    parser.add_argument("--bce-label-smoothing", type=float, default=None, help="Label smoothing factor for the classification head in the two_head model")
    parser.add_argument("--tree-max-iter", type=int, default=None, help="Number of boosting iterations for tree-based models")
    parser.add_argument("--tree-learning-rate", type=float, default=None, help="Learning rate for tree-based models")
    parser.add_argument("--tree-max-depth", type=int, default=None, help="Maximum depth for tree-based models")
    parser.add_argument("--tree-l2", type=float, default=None, help="L2 regularization strength for tree-based models")
    parser.add_argument("--tree-subsample", type=float, default=None, help="Subsample ratio for tree-based models")
    parser.add_argument("--tree-min-samples-leaf", type=int, default=None, help="Minimum samples per leaf for tree-based models")
    parser.add_argument("--tree-max-bins", type=int, default=None, help="Maximum number of histogram bins for tree-based models")
    parser.add_argument("--lgbm-n-estimators", type=int, default=None, help="Number of boosting iterations for LightGBM")
    parser.add_argument("--lgbm-learning-rate", type=float, default=None, help="Learning rate for LightGBM")
    parser.add_argument("--lgbm-num-leaves", type=int, default=None, help="Maximum leaves per tree for LightGBM")
    parser.add_argument("--lgbm-feature-fraction", type=float, default=None, help="Column sampling ratio for LightGBM")
    parser.add_argument("--lgbm-bagging-fraction", type=float, default=None, help="Row sampling ratio for LightGBM")
    parser.add_argument("--lgbm-bagging-freq", type=int, default=None, help="Bagging frequency for LightGBM")
    parser.add_argument("--lgbm-lambda-l2", type=float, default=None, help="L2 regularization strength for LightGBM")
    parser.add_argument("--lgbm-min-child-samples", type=int, default=None, help="Minimum child samples for LightGBM leaves")
    parser.add_argument("--lgbm-max-depth", type=int, default=None, help="Maximum depth for LightGBM trees (-1 for unlimited)")
    parser.add_argument("--lgbm-sample-fraction", type=float, default=None, help="Row sampling ratio applied before LightGBM training")
    parser.add_argument("--lgbm-early-stopping", type=int, default=None, help="Early stopping rounds for LightGBM")
    parser.add_argument("--lgbm-log-period", type=int, default=None, help="Log evaluation metric every n rounds for LightGBM (0 to disable)")
    parser.add_argument("--inference-only", action="store_true", help="Skip training and run inference using a saved checkpoint")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Single checkpoint file to load when running in inference-only mode")
    parser.add_argument(
        "--checkpoint-paths",
        type=Path,
        action="append",
        default=None,
        help="Repeatable option to supply multiple checkpoint files for ensemble inference",
    )
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
    if args.model is not None:
        config.model_name = args.model
    if args.pv_embedding_dim is not None:
        config.pv_embedding_dim = args.pv_embedding_dim
    if args.bce_weight is not None:
        config.bce_weight = args.bce_weight
    if args.reg_weight is not None:
        config.regression_weight = args.reg_weight
    if args.positive_threshold is not None:
        config.positive_threshold = args.positive_threshold
    if args.bce_label_smoothing is not None:
        config.bce_label_smoothing = args.bce_label_smoothing
    if args.tree_max_iter is not None:
        config.tree_max_iter = args.tree_max_iter
    if args.tree_learning_rate is not None:
        config.tree_learning_rate = args.tree_learning_rate
    if args.tree_max_depth is not None:
        config.tree_max_depth = args.tree_max_depth
    if args.tree_l2 is not None:
        config.tree_l2_regularization = args.tree_l2
    if args.tree_subsample is not None:
        config.tree_subsample = args.tree_subsample
    if args.tree_min_samples_leaf is not None:
        config.tree_min_samples_leaf = args.tree_min_samples_leaf
    if args.tree_max_bins is not None:
        config.tree_max_bins = args.tree_max_bins
    if args.lgbm_n_estimators is not None:
        config.lgbm_n_estimators = args.lgbm_n_estimators
    if args.lgbm_learning_rate is not None:
        config.lgbm_learning_rate = args.lgbm_learning_rate
    if args.lgbm_num_leaves is not None:
        config.lgbm_num_leaves = args.lgbm_num_leaves
    if args.lgbm_feature_fraction is not None:
        config.lgbm_feature_fraction = args.lgbm_feature_fraction
    if args.lgbm_bagging_fraction is not None:
        config.lgbm_bagging_fraction = args.lgbm_bagging_fraction
    if args.lgbm_bagging_freq is not None:
        config.lgbm_bagging_freq = args.lgbm_bagging_freq
    if args.lgbm_lambda_l2 is not None:
        config.lgbm_lambda_l2 = args.lgbm_lambda_l2
    if args.lgbm_min_child_samples is not None:
        config.lgbm_min_child_samples = args.lgbm_min_child_samples
    if args.lgbm_max_depth is not None:
        config.lgbm_max_depth = args.lgbm_max_depth
    if args.lgbm_sample_fraction is not None:
        config.lgbm_sample_fraction = args.lgbm_sample_fraction
    if args.lgbm_early_stopping is not None:
        config.lgbm_early_stopping_rounds = args.lgbm_early_stopping
    if args.lgbm_log_period is not None:
        config.lgbm_log_evaluation_period = args.lgbm_log_period
    if args.inference_only:
        config.inference_only = True
    if args.checkpoint_path is not None:
        config.checkpoint_path = args.checkpoint_path
    if args.checkpoint_paths:
        config.checkpoint_paths = tuple(args.checkpoint_paths)


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    apply_overrides(config, args)
    logger.info(f"Starting pipeline with config: {config}")
    run_pipeline(config)


if __name__ == "__main__":
    main()
