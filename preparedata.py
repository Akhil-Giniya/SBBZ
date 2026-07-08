"""
Prepares raw bazaar Parquet data for ML training:
- loads every file collected by bazaar_collector.py
- sorts by product then timestamp (required for correct lag features)
- adds lag features (price N steps ago) per product
- adds a rolling average as a simple trend feature
- saves the result as a single training-ready Parquet file

Run this AFTER you've collected data, not as part of the live collector.

Usage:
    pip install pandas pyarrow
    python prepare_training_data.py
"""

import logging
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/skyblock_bazaar")
OUTPUT_PATH = Path("data/training_dataset.parquet")

# How many polls back to look for lag features.
# With a 60s poll interval: 1 = 1 min ago, 5 = 5 min ago, 60 = 1 hour ago.
LAG_STEPS = [1, 5, 15, 60]
ROLLING_WINDOW = 15  # ~15 min rolling average, given 60s polling

# --- Target: forward-looking flip profit margin ---
# HORIZON_STEPS: how far ahead we're predicting (15 steps * 60s poll = 15 min).
# BAZAAR_TAX: Hypixel's bazaar sell tax as a fraction -- VERIFY this against
# current game mechanics, it's a placeholder and may be out of date / may
# vary by Bazaar Tax Reduction perk level.
HORIZON_STEPS = 15
BAZAAR_TAX = 0.0125

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("prepare_training_data.log"), logging.StreamHandler()],
)
log = logging.getLogger("prepare_training_data")


def main():
    log.info(f"Loading raw data from {DATA_DIR} ...")
    df = pd.read_parquet(DATA_DIR)
    log.info(f"Loaded {len(df):,} rows.")

    # This is the step that has to happen at load time, not collection time:
    # group by product so lag/rolling features don't leak across products,
    # and sort by timestamp within each product so "1 step ago" is meaningful.
    df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)

    grouped = df.groupby("product", group_keys=False)

    for lag in LAG_STEPS:
        df[f"mid_price_lag_{lag}"] = grouped["mid_price"].shift(lag)
        df[f"spread_lag_{lag}"] = grouped["spread"].shift(lag)
        df[f"buy_volume_lag_{lag}"] = grouped["buy_volume"].shift(lag)
        df[f"sell_volume_lag_{lag}"] = grouped["sell_volume"].shift(lag)
        df[f"demand_supply_ratio_lag_{lag}"] = grouped["demand_supply_ratio"].shift(lag)

    df["mid_price_rolling_mean"] = grouped["mid_price"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["mid_price_rolling_std"] = grouped["mid_price"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=2).std()
    )

    # A buy_price of 0 or negative is invalid data (API glitch, not a real
    # price) -- drop these rows rather than dividing by a near-zero epsilon,
    # which would produce a corrupted target value in the billions and
    # poison training on that row instead of safely avoiding the problem.
    invalid_price_mask = df["buy_price"] <= 0
    if invalid_price_mask.any():
        log.warning(f"Dropping {invalid_price_mask.sum()} rows with invalid buy_price <= 0")
        df = df[~invalid_price_mask].reset_index(drop=True)
        grouped = df.groupby("product", group_keys=False)  # regroup after filtering

    # Target: if you bought at today's buy_price and sold HORIZON_STEPS later
    # at whatever sell_price is then, what margin would you have made, net
    # of tax? This is what the model will learn to predict from CURRENT
    # features (it never sees future_sell_price directly -- only the target
    # derived from it).
    future_sell_price = grouped["sell_price"].shift(-HORIZON_STEPS)
    df["profit_margin_target"] = (
        (future_sell_price - df["buy_price"]) / df["buy_price"]
    ) - BAZAAR_TAX

    # Rows at the very start of each product's history won't have full lag
    # history yet, and rows at the very end won't have future data yet for
    # the target -- drop both so the model doesn't train on incomplete rows.
    # (Using cumcount-based masks here rather than groupby().apply() with
    # slicing -- that approach silently drops the group column in some
    # pandas versions.)
    max_lag = max(LAG_STEPS)
    pos_from_start = df.groupby("product").cumcount()
    pos_from_end = df.groupby("product").cumcount(ascending=False)
    mask = (pos_from_start >= max_lag) & (pos_from_end >= HORIZON_STEPS)
    df = df[mask].reset_index(drop=True)

    # Safety net for NaNs the position-based trim above doesn't cover --
    # e.g. Hypixel's API omits buy_moving_week/sell_moving_week for some
    # products entirely, which the collector stores as None regardless of
    # row position.
    before_dropna = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before_dropna - len(df)
    if dropped:
        log.warning(f"Dropped {dropped} additional rows with missing values (e.g. moving_week fields).")

    # Optional classification target, for later: "will this flip be
    # profitable at all" as a yes/no, alongside the regression target's
    # "how much profit". Computed only after dropna/trimming above, so it's
    # never derived from a still-undefined profit_margin_target.
    df["is_profitable"] = (df["profit_margin_target"] > 0).astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, compression="snappy", index=False)
    log.info(f"Saved {len(df):,} training-ready rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()