"""
Prepares raw bazaar Parquet data for ML training:
- Loads daily Parquet files collected by dataloop.py (ignoring SQLite buffers)
- Asserts configuration invariants to prevent feature corruption
- Filters out invalid/corrupted buy and sell prices early
- Tracks comprehensive product lineage (attrition from prices, trimming, and NaNs)
- Sorts by product then timestamp
- Logs time-series gap diagnostics to verify API polling regularity
- Adds lag features and exact time elapsed per product
- Adds expanded rolling and EMA features
- Safely trims series edges and logs attrition (insufficient history)
- Fails fast if no valid data remains
- Saves the result and metadata for reproducible training
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ---------------- Configuration ----------------

@dataclass
class PipelineConfig:
    data_dir: Path = Path("data/skyblock_bazaar")
    output_parquet: Path = Path("data/training_dataset.parquet")
    output_metadata: Path = Path("data/training_metadata.json")

    # Lag steps represent ROWS ago, not necessarily minutes ago.
    lag_steps: list[int] = field(default_factory=lambda: [1, 5, 15, 60])

    # Window size for rolling statistics
    rolling_window: int = 15

    # Target variables
    horizon_steps: int = 15
    bazaar_tax: float = 0.0125

config = PipelineConfig()

# Initialize logging before invariant checks so fatal config errors are actually captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("prepare_training_data.log"), logging.StreamHandler()],
)
log = logging.getLogger("prepare_training_data")

# Invariant check: Prevent rolling stats from leaking NaNs into the final output
if config.rolling_window > max(config.lag_steps):
    error_msg = (
        f"Configuration Error: rolling_window ({config.rolling_window}) cannot "
        f"exceed max(lag_steps) ({max(config.lag_steps)}). Trimming will fail to cover the rolling window."
    )
    log.error(error_msg)
    raise ValueError(error_msg)

# ---------------- Main Pipeline ----------------

def main():
    log.info(f"Searching for daily Parquet files in {config.data_dir} ...")

    files = sorted(config.data_dir.glob("*/daily_bazaar.parquet"))
    if not files:
        log.error(f"No daily_bazaar.parquet files found under {config.data_dir}")
        raise FileNotFoundError(f"No daily_bazaar.parquet files found under {config.data_dir}")

    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    log.info(f"Loaded {len(df):,} rows.")

    # Absolute initial state for lineage tracking
    products_initial = set(df["product"].unique())

    # 1. Early Filtering of Invalid Data
    # MUST happen before lag/rolling calculations so dropped rows don't leave "phantom lags".
    invalid_price_mask = (
        df["buy_price"].isna() | (df["buy_price"] <= 0) |
        df["sell_price"].isna() | (df["sell_price"] <= 0)
    )
    if invalid_price_mask.any():
        log.warning(f"Dropping {invalid_price_mask.sum():,} rows with missing/invalid buy_price or sell_price <= 0")
        df = df[~invalid_price_mask].reset_index(drop=True)

    products_after_filter = set(df["product"].unique())
    dropped_by_price = products_initial - products_after_filter

    if dropped_by_price:
        sample_price_dropped = sorted(list(dropped_by_price))[:5]
        log.warning(
            f"Completely excluded {len(dropped_by_price)} products due to entirely invalid/missing prices "
            f"(e.g., {', '.join(sample_price_dropped)})."
        )

    # Group by product so lag/rolling features don't leak across products
    df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)

    # Diagnostic: Measure actual time gaps between recorded rows (measured on clean data)
    gaps = df.groupby("product")["timestamp"].diff().dt.total_seconds().dropna()
    if not gaps.empty:
        log.info(
            f"Row gap diagnostics - Median: {gaps.median():.0f}s, "
            f"95th pct: {gaps.quantile(0.95):.0f}s, Max: {gaps.max():.0f}s"
        )
    else:
        log.info("Row gap diagnostics - Insufficient data to measure consecutive row gaps.")

    grouped = df.groupby("product", group_keys=False)

    # 2. Lag Features (Row-based, time-aware)
    for lag in config.lag_steps:
        df[f"mid_price_lag_{lag}"] = grouped["mid_price"].shift(lag)
        df[f"spread_lag_{lag}"] = grouped["spread"].shift(lag)
        df[f"buy_volume_lag_{lag}"] = grouped["buy_volume"].shift(lag)
        df[f"sell_volume_lag_{lag}"] = grouped["sell_volume"].shift(lag)
        df[f"demand_supply_ratio_lag_{lag}"] = grouped["demand_supply_ratio"].shift(lag)

        # Give the model the actual time elapsed so it can adjust for irregular sampling
        df[f"time_elapsed_lag_{lag}"] = grouped["timestamp"].diff(lag).dt.total_seconds()

    # 3. Expanded Rolling Features
    df["mid_price_rolling_mean"] = grouped["mid_price"].transform(
        lambda s: s.rolling(config.rolling_window, min_periods=1).mean()
    )
    df["mid_price_rolling_std"] = grouped["mid_price"].transform(
        lambda s: s.rolling(config.rolling_window, min_periods=2).std()
    )
    df["mid_price_rolling_max"] = grouped["mid_price"].transform(
        lambda s: s.rolling(config.rolling_window, min_periods=1).max()
    )
    df["mid_price_rolling_min"] = grouped["mid_price"].transform(
        lambda s: s.rolling(config.rolling_window, min_periods=1).min()
    )
    df["mid_price_rolling_median"] = grouped["mid_price"].transform(
        lambda s: s.rolling(config.rolling_window, min_periods=1).median()
    )
    df["mid_price_ema"] = grouped["mid_price"].transform(
        lambda s: s.ewm(span=config.rolling_window, adjust=False).mean()
    )

    # 4. Target Generation
    future_sell_price = grouped["sell_price"].shift(-config.horizon_steps)
    df["profit_margin_target"] = (
        (future_sell_price - df["buy_price"]) / df["buy_price"]
    ) - config.bazaar_tax

    df["target_time_horizon_sec"] = grouped["timestamp"].diff(-config.horizon_steps).dt.total_seconds().abs()

    # 5. Safe Trimming & Series Edge Logging
    max_lag = max(config.lag_steps)
    pos_from_start = df.groupby("product").cumcount()
    pos_from_end = df.groupby("product").cumcount(ascending=False)
    mask = (pos_from_start >= max_lag) & (pos_from_end >= config.horizon_steps)

    before_trim = len(df)

    df = df[mask].reset_index(drop=True)

    trimmed = before_trim - len(df)
    if trimmed:
        log.info(f"Trimmed {trimmed:,} rows from series edges (insufficient lag/horizon history).")

    products_after_trim = set(df["product"].unique())
    fully_excluded_by_trim = products_after_filter - products_after_trim

    if fully_excluded_by_trim:
        sample_trim_dropped = sorted(list(fully_excluded_by_trim))[:5]
        log.warning(
            f"{len(fully_excluded_by_trim)} products had insufficient history to survive "
            f"trimming and were dropped entirely (e.g., {', '.join(sample_trim_dropped)})."
        )

    # 6. Targeted dropna & Missing Feature Attrition Logging
    before_dropna = len(df)

    required_cols = [
        "profit_margin_target", "target_time_horizon_sec", "buy_price", "sell_price", "mid_price", "spread"
    ]
    for lag in config.lag_steps:
        required_cols.extend([
            f"mid_price_lag_{lag}",
            f"spread_lag_{lag}",
            f"buy_volume_lag_{lag}",
            f"sell_volume_lag_{lag}",
            f"demand_supply_ratio_lag_{lag}", # Closed validation gap
            f"time_elapsed_lag_{lag}"
        ])

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    dropped_rows = before_dropna - len(df)
    products_after_dropna = set(df["product"].unique())
    dropped_products_na = products_after_trim - products_after_dropna

    if dropped_rows:
        log.info(f"Dropped {dropped_rows:,} rows missing core required features.")
    if dropped_products_na:
        sample_dropped_na = sorted(list(dropped_products_na))[:5]
        log.warning(
            f"Completely excluded {len(dropped_products_na)} illiquid products due to perpetual NaNs "
            f"(e.g., {', '.join(sample_dropped_na)})."
        )

    # Fail fast if data is completely exhausted
    if df.empty:
        log.error("Fatal: No training rows remain after feature generation and filtering.")
        raise RuntimeError(
            "Fatal: No training rows remain after feature generation and filtering. "
            "You likely need to collect more historical data to satisfy the required lag/horizon steps."
        )

    # 7. Type Casting & Memory Optimization
    df["is_profitable"] = (df["profit_margin_target"] > 0).astype("int8")

    if "hour" in df.columns:
        df["hour"] = df["hour"].astype("int8")
    if "day_of_week" in df.columns:
        df["day_of_week"] = df["day_of_week"].astype("int8")

    for lag in config.lag_steps:
        df[f"buy_volume_lag_{lag}"] = df[f"buy_volume_lag_{lag}"].astype("int64")
        df[f"sell_volume_lag_{lag}"] = df[f"sell_volume_lag_{lag}"].astype("int64")
        df[f"time_elapsed_lag_{lag}"] = df[f"time_elapsed_lag_{lag}"].astype("int32")

    df["target_time_horizon_sec"] = df["target_time_horizon_sec"].astype("int32")

    # 8. Save Data
    config.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.output_parquet, compression="snappy", index=False)
    log.info(f"Saved {len(df):,} training-ready rows to {config.output_parquet}")

    # 9. Save Feature Metadata
    metadata = asdict(config)
    metadata["data_dir"] = str(metadata["data_dir"])
    metadata["output_parquet"] = str(metadata["output_parquet"])
    metadata["output_metadata"] = str(metadata["output_metadata"])
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()

    # Global state tracking
    metadata["total_rows"] = len(df)
    metadata["total_products_initial"] = len(products_initial)
    metadata["total_products_kept"] = len(products_after_dropna)

    # Deterministic metadata for Invalid Price Attrition
    metadata["dropped_by_price_count"] = len(dropped_by_price)
    metadata["dropped_by_price_sample"] = sorted(list(dropped_by_price))[:10] if dropped_by_price else []

    # Deterministic metadata for Trim Attrition
    metadata["dropped_by_trim_count"] = len(fully_excluded_by_trim)
    metadata["dropped_by_trim_sample"] = sorted(list(fully_excluded_by_trim))[:10] if fully_excluded_by_trim else []

    # Deterministic metadata for NaN Attrition
    metadata["dropped_by_nan_count"] = len(dropped_products_na)
    metadata["dropped_by_nan_sample"] = sorted(list(dropped_products_na))[:10] if dropped_products_na else []

    config.output_metadata.parent.mkdir(parents=True, exist_ok=True)

    with open(config.output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    log.info(f"Saved feature metadata to {config.output_metadata}")

if __name__ == "__main__":
    main()