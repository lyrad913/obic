from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from loguru import logger


NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


@dataclass(slots=True)
class DataBundle:
    feature_names: Sequence[str]
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    train_meta: pl.DataFrame
    val_meta: pl.DataFrame
    test_meta: pl.DataFrame
    submission_template: pl.DataFrame
    scaler: StandardScaler | None
    train_pv_indices: np.ndarray | None = None
    val_pv_indices: np.ndarray | None = None
    test_pv_indices: np.ndarray | None = None
    pv_id_mapping: dict[str, int] | None = None
    train_frame: pl.DataFrame | None = None
    val_frame: pl.DataFrame | None = None
    test_frame: pl.DataFrame | None = None
    categorical_features: Sequence[str] | None = None


def load_raw_frames(data_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    logger.info(f"CSV 로부터 데이터를 로딩합니다. data_dir={data_dir}")
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    submission_path = data_dir / "submission_sample.csv"

    train_df = pl.read_csv(train_path, try_parse_dates=True)
    test_df = pl.read_csv(test_path, try_parse_dates=True)
    submission_df = pl.read_csv(submission_path, try_parse_dates=True)
    logger.info(f"데이터 로딩 완료: train={train_df.height} rows, test={test_df.height} rows")
    return train_df, test_df, submission_df


def ensure_datetime(df: pl.DataFrame, column: str) -> pl.DataFrame:
    dtype = df.schema[column]
    if hasattr(dtype, "is_temporal") and dtype.is_temporal():
        return df
    return df.with_columns(pl.col(column).str.to_datetime(strict=False).alias(column))


def add_time_features(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    df = ensure_datetime(df, time_col)
    two_pi = 2.0 * np.pi
    return df.with_columns(
        [
            pl.col(time_col).dt.hour().alias("hour"),
            pl.col(time_col).dt.minute().alias("minute"),
            pl.col(time_col).dt.day().alias("day"),
            pl.col(time_col).dt.month().alias("month"),
            pl.col(time_col).dt.weekday().alias("weekday"),
            pl.col(time_col).dt.ordinal_day().alias("day_of_year"),
        ]
    ).with_columns(
        [
            (pl.col("hour") * two_pi / 24.0).sin().alias("hour_sin"),
            (pl.col("hour") * two_pi / 24.0).cos().alias("hour_cos"),
            (pl.col("day_of_year") * two_pi / 365.0).sin().alias("day_of_year_sin"),
            (pl.col("day_of_year") * two_pi / 365.0).cos().alias("day_of_year_cos"),
            ((pl.col("month") - 1) * two_pi / 12.0).sin().alias("month_sin"),
            ((pl.col("month") - 1) * two_pi / 12.0).cos().alias("month_cos"),
        ]
    )


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    derived_columns = []
    if "temp_a" in df.columns and "temp_b" in df.columns:
        derived_columns.append((pl.col("temp_a") - pl.col("temp_b")).alias("temp_diff_ab"))
    if "wind_spd_a" in df.columns and "wind_spd_b" in df.columns:
        derived_columns.append((pl.col("wind_spd_a") - pl.col("wind_spd_b")).alias("wind_spd_diff"))
    if "pressure" in df.columns and "ground_press" in df.columns:
        derived_columns.append((pl.col("pressure") - pl.col("ground_press")).alias("pressure_gradient"))
    if "humidity" in df.columns and "rel_hum" in df.columns:
        derived_columns.append((pl.col("humidity") - pl.col("rel_hum")).alias("humidity_gap"))
    if not derived_columns:
        return df
    return df.with_columns(derived_columns)


def fill_missing_by_group(
    df: pl.DataFrame,
    group_col: str,
    time_col: str,
    fill_cols: Iterable[str],
) -> pl.DataFrame:
    df = df.sort([group_col, time_col])
    fill_cols = list(fill_cols)
    forward_expr = [pl.col(col).fill_null(strategy="forward").over(group_col) for col in fill_cols]
    df = df.with_columns(forward_expr)
    backward_expr = [pl.col(col).fill_null(strategy="backward").over(group_col) for col in fill_cols]
    df = df.with_columns(backward_expr)
    if fill_cols:
        means_df = df.select([pl.col(col).mean().alias(col) for col in fill_cols])
        mean_values = {}
        for col in fill_cols:
            value = means_df[col][0]
            if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
                value = 0.0
            mean_values[col] = value
        fallback_expr = [pl.col(col).fill_null(mean_values[col]).alias(col) for col in fill_cols]
        df = df.with_columns(fallback_expr)
    return df


def select_numerical_columns(df: pl.DataFrame, exclude: Iterable[str]) -> list[str]:
    return [
        col
        for col, dtype in df.schema.items()
        if col not in exclude and dtype in NUMERIC_DTYPES
    ]


def preprocess_frames(config: TrainingConfig) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int]]:
    logger.info("전처리를 시작합니다.")
    train_df, test_df, submission_df = load_raw_frames(config.data_dir)
    train_df = add_time_features(train_df, config.time_col)
    test_df = add_time_features(test_df, config.time_col)
    logger.debug("시간 파생 특성을 추가했습니다.")

    train_df = add_interaction_features(train_df)
    test_df = add_interaction_features(test_df)
    logger.debug("상호작용 특성을 추가했습니다.")

    if "energy" in train_df.columns:
        train_df = train_df.drop("energy")

    train_df = train_df.filter(pl.col(config.target_col).is_not_null())
    logger.info(f"타깃 결측 행 제거 후 train={train_df.height} rows")

    pv_union = pl.concat([
        train_df.select(config.group_col).unique(),
        test_df.select(config.group_col).unique(),
    ]).unique()
    pv_vocab = sorted(pv_union.select(config.group_col).to_series().to_list())
    pv_mapping = {pv: idx for idx, pv in enumerate(pv_vocab)}

    train_df = train_df.with_columns(
        pl.col(config.group_col).replace(pv_mapping).cast(pl.Int32).alias("pv_idx")
    )
    test_df = test_df.with_columns(
        pl.col(config.group_col).replace(pv_mapping).cast(pl.Int32).alias("pv_idx")
    )

    feature_exclude = {config.target_col, config.group_col, config.time_col, "type"}
    if not config.is_tree_model:
        feature_exclude.add("pv_idx")
    feature_columns = select_numerical_columns(train_df, feature_exclude)
    if config.is_tree_model and "pv_idx" not in feature_columns and "pv_idx" in train_df.columns:
        feature_columns.append("pv_idx")
    feature_columns = sorted(feature_columns)
    logger.info(f"선택된 피처 수: {len(feature_columns)}")

    train_df = fill_missing_by_group(train_df, config.group_col, config.time_col, feature_columns)
    test_df = fill_missing_by_group(test_df, config.group_col, config.time_col, feature_columns)
    logger.info("결측치 보간을 완료했습니다.")

    train_cast_expr = []
    test_cast_expr = []
    for col in feature_columns:
        dtype = pl.Int32 if col == "pv_idx" else pl.Float32
        train_cast_expr.append(pl.col(col).cast(dtype))
        test_cast_expr.append(pl.col(col).cast(dtype))
    train_df = train_df.with_columns(train_cast_expr + [pl.col(config.target_col).cast(pl.Float32)])
    test_df = test_df.with_columns(test_cast_expr)
    logger.debug("데이터 타입 캐스팅 완료.")

    return train_df, test_df, submission_df, pv_mapping


def split_train_validation(
    train_df: pl.DataFrame,
    config: TrainingConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    group_series = train_df.select(config.group_col).to_series()
    unique_groups = group_series.unique().to_numpy()
    if unique_groups.size < 2:
        raise ValueError("Need at least two unique groups to create a validation split.")
    test_size = int(round(unique_groups.size * config.val_ratio))
    test_size = max(1, min(test_size, unique_groups.size - 1))
    rng = np.random.default_rng(config.seed)
    shuffled_groups = rng.permutation(unique_groups)
    val_groups = list(shuffled_groups[:test_size])
    train_split = train_df.filter(pl.col(config.group_col).is_in(val_groups).not_())
    val_split = train_df.filter(pl.col(config.group_col).is_in(val_groups))
    return train_split, val_split


def to_numpy(df: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return df.select(columns).to_numpy().astype(np.float32, copy=False)


def prepare_data(config: TrainingConfig) -> DataBundle:
    train_df, test_df, submission_df, pv_mapping = preprocess_frames(config)
    feature_exclude = {config.target_col, config.group_col, config.time_col, "type"}
    if not config.is_tree_model:
        feature_exclude.add("pv_idx")
    feature_columns = select_numerical_columns(train_df, feature_exclude)
    if config.is_tree_model and "pv_idx" not in feature_columns and "pv_idx" in train_df.columns:
        feature_columns.append("pv_idx")
    feature_columns = sorted(feature_columns)
    categorical_features: list[str] = []
    if config.is_tree_model and "pv_idx" in feature_columns:
        categorical_features.append("pv_idx")

    train_split, val_split = split_train_validation(train_df, config)
    train_groups = train_split.select(config.group_col).n_unique()
    val_groups = val_split.select(config.group_col).n_unique()
    logger.info(
        f"검증 세트 분할 완료: train_groups={train_groups}, val_groups={val_groups}, train_rows={train_split.height}, val_rows={val_split.height}"
    )

    train_features_frame = train_split.select(feature_columns)
    val_features_frame = val_split.select(feature_columns)
    test_features_frame = test_df.select(feature_columns)

    train_array = train_features_frame.to_numpy()
    val_array = val_features_frame.to_numpy()
    test_array = test_features_frame.to_numpy()

    scaler: StandardScaler | None = None
    if config.scale_features:
        scaler = StandardScaler()
        logger.info("스케일링을 시작합니다 (StandardScaler).")
        x_train = scaler.fit_transform(train_array.astype(np.float32, copy=False))
        x_val = scaler.transform(val_array.astype(np.float32, copy=False))
        x_test = scaler.transform(test_array.astype(np.float32, copy=False))
        logger.info("스케일링을 완료했습니다.")
    else:
        x_train = train_array.astype(np.float32, copy=False)
        x_val = val_array.astype(np.float32, copy=False)
        x_test = test_array.astype(np.float32, copy=False)
        logger.info("스케일링을 생략하고 원본 피처를 사용합니다.")

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

    train_pv_indices = train_split.select("pv_idx").to_numpy().astype(np.int64, copy=False).ravel()
    val_pv_indices = val_split.select("pv_idx").to_numpy().astype(np.int64, copy=False).ravel()
    test_pv_indices = test_df.select("pv_idx").to_numpy().astype(np.int64, copy=False).ravel()

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
        scaler=scaler,
        train_pv_indices=train_pv_indices,
        val_pv_indices=val_pv_indices,
        test_pv_indices=test_pv_indices,
        pv_id_mapping=pv_mapping,
        train_frame=train_features_frame if config.is_tree_model else None,
        val_frame=val_features_frame if config.is_tree_model else None,
        test_frame=test_features_frame if config.is_tree_model else None,
        categorical_features=tuple(categorical_features) if categorical_features else None,
    )
