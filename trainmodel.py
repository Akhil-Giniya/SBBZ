"""
Trains a baseline model to predict flip profit margin per product, and
evaluates it the way that actually matters here: not just raw prediction
error, but "if I used this model's top picks, would I have made money?"

Run this AFTER prepare_training_data.py.

Usage:
    pip install scikit-learn joblib pandas pyarrow
    python train_model.py
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, precision_score, recall_score

DATASET_PATH = Path("data/training_dataset.parquet")
MODEL_PATH = Path("data/flip_model.joblib")

# Fraction of the time range held out as a test set. This MUST be a time
# cutoff, not a random split -- a random split would let the model "see"
# rows from the future during training, which silently inflates accuracy
# and produces a model that looks great here and fails in production.
TEST_FRACTION = 0.15

# Columns the model is not allowed to see: identifiers, and anything that
# depends on the future (both target variants -- is_profitable is derived
# from profit_margin_target, so it leaks the answer just as badly if left
# in as a feature).
NON_FEATURE_COLUMNS = ["timestamp", "product", "profit_margin_target", "is_profitable"]

# --- Trading realism config ---
# MIN_PROFIT_THRESHOLD: don't trade at all if nothing clears this bar, even
# if it's the "best available" option -- the best of a bad set is still bad.
# HOLDING_PERIOD_MINUTES: must match HORIZON_STEPS in prepare_training_data.py
# (15 steps * 60s poll = 15 min) -- this is how long a position's capital
# stays locked up before it can be reinvested.
MIN_PROFIT_THRESHOLD = 0.01
PROB_THRESHOLD = 0.5  # classifier's minimum "this will be profitable" confidence to trade at all
HOLDING_PERIOD_MINUTES = 15
POSITION_SIZE = 1_000.0
MAX_CONCURRENT_POSITIONS = 10
STARTING_CAPITAL = 10_000.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("train_model.log"), logging.StreamHandler()],
)
log = logging.getLogger("train_model")


def time_based_split(df: pd.DataFrame):
    df = df.sort_values("timestamp")
    cutoff_index = int(len(df) * (1 - TEST_FRACTION))
    cutoff_time = df.iloc[cutoff_index]["timestamp"]
    train = df[df["timestamp"] < cutoff_time]
    test = df[df["timestamp"] >= cutoff_time]
    return train, test, cutoff_time


def evaluate_ranking(test_df: pd.DataFrame, predictions: np.ndarray, top_k: int = 5,
                      min_profit_threshold: float = MIN_PROFIT_THRESHOLD,
                      classifier_proba: np.ndarray = None,
                      prob_threshold: float = PROB_THRESHOLD):
    """
    At each timestamp in the test set, rank products by PREDICTED profit
    margin and take the top_k -- but only among candidates that clear
    min_profit_threshold (and, if classifier_proba is given, that the
    classifier is also confident enough is profitable). If nothing clears
    the bar at a given timestamp, no trade is taken there at all.

    Hybrid approach: the classifier answers "is this worth trading at all"
    (filter), the regressor answers "which of the qualifying ones is best"
    (rank). Passing classifier_proba=None falls back to regressor-only
    filtering, same as before.
    """
    eval_df = test_df.copy()
    eval_df["predicted"] = predictions
    if classifier_proba is not None:
        eval_df["predicted_proba"] = classifier_proba

    picked_profits = []
    baseline_profits = []
    timestamps_with_no_qualifying_trade = 0
    total_trades_taken = 0

    for _, group in eval_df.groupby("timestamp"):
        baseline_profits.append(group["profit_margin_target"].mean())

        qualifies = group["predicted"] > min_profit_threshold
        if classifier_proba is not None:
            qualifies &= group["predicted_proba"] > prob_threshold
        candidates = group[qualifies]

        if candidates.empty:
            timestamps_with_no_qualifying_trade += 1
            continue

        top_picks = candidates.nlargest(top_k, "predicted")
        picked_profits.append(top_picks["profit_margin_target"].mean())
        total_trades_taken += len(top_picks)

    return {
        "avg_profit_margin_of_top_picks": float(np.mean(picked_profits)) if picked_profits else None,
        "avg_profit_margin_random_baseline": float(np.mean(baseline_profits)),
        "num_timestamps_with_trades": len(picked_profits),
        "num_timestamps_with_no_qualifying_trade": timestamps_with_no_qualifying_trade,
        "total_trades_taken": total_trades_taken,
    }


def realistic_backtest(test_df: pd.DataFrame, predictions: np.ndarray,
                        holding_period_minutes: float = HOLDING_PERIOD_MINUTES,
                        min_profit_threshold: float = MIN_PROFIT_THRESHOLD,
                        position_size: float = POSITION_SIZE,
                        max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS,
                        starting_capital: float = STARTING_CAPITAL,
                        classifier_proba: np.ndarray = None,
                        prob_threshold: float = PROB_THRESHOLD):
    """
    Walks through the test set in chronological order with a fixed pool of
    capital, simulating what would have actually happened: capital used to
    open a position is locked up until holding_period_minutes later, only
    max_concurrent_positions can be open at once, and a trade is only taken
    if there's capital free to take it, and (if classifier_proba is given)
    only if the classifier is confident enough it'll be profitable at all.

    Also tracks an equity curve (account value over time: free cash + capital
    still tied up in open positions) and the max drawdown from that curve --
    the biggest peak-to-trough dip, which an average return figure hides
    completely. A strategy can have a good average return and still have
    wiped out most of its capital at some point along the way; drawdown is
    what surfaces that.

    Note: this assumes you always get filled at the target price and can
    always find a counterparty at that price -- it doesn't model order-book
    competition. Treat the output as an upper bound on realistic returns,
    not a guarantee.
    """
    eval_df = test_df.copy()
    eval_df["predicted"] = predictions
    if classifier_proba is not None:
        eval_df["predicted_proba"] = classifier_proba
    eval_df = eval_df.sort_values("timestamp")

    capital = starting_capital
    open_positions = []  # each: {close_time, capital_committed, profit_margin}
    trade_log = []
    equity_curve = []  # (timestamp, account_value)

    for timestamp, group in eval_df.groupby("timestamp", sort=True):
        # Free up capital from positions that have matured by now.
        still_open = []
        for pos in open_positions:
            if pos["close_time"] <= timestamp:
                capital += pos["capital_committed"] * (1 + pos["profit_margin"])
            else:
                still_open.append(pos)
        open_positions = still_open

        qualifies = group["predicted"] > min_profit_threshold
        if classifier_proba is not None:
            qualifies &= group["predicted_proba"] > prob_threshold
        candidates = group[qualifies].sort_values("predicted", ascending=False)

        for _, row in candidates.iterrows():
            if len(open_positions) >= max_concurrent_positions or capital < position_size:
                break
            capital -= position_size
            open_positions.append({
                "close_time": timestamp + pd.Timedelta(minutes=holding_period_minutes),
                "capital_committed": position_size,
                "profit_margin": row["profit_margin_target"],
            })
            trade_log.append(row["profit_margin_target"])

        account_value = capital + sum(p["capital_committed"] for p in open_positions)
        equity_curve.append((timestamp, account_value))

    # Settle whatever's still open at the end of the test window.
    for pos in open_positions:
        capital += pos["capital_committed"] * (1 + pos["profit_margin"])

    equity_series = pd.Series([v for _, v in equity_curve])
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown_pct = float(drawdown.min()) if len(drawdown) else 0.0

    return {
        "starting_capital": starting_capital,
        "final_capital": capital,
        "total_return_pct": (capital - starting_capital) / starting_capital,
        "total_trades": len(trade_log),
        "avg_profit_margin_per_trade": float(np.mean(trade_log)) if trade_log else None,
        "max_drawdown_pct": max_drawdown_pct,
        "equity_curve": equity_curve,  # kept for later charting in the future dashboard
    }


def main():
    log.info(f"Loading {DATASET_PATH} ...")
    df = pd.read_parquet(DATASET_PATH)
    log.info(f"Loaded {len(df):,} rows.")

    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    log.info(f"Using {len(feature_columns)} features: {feature_columns}")

    train_df, test_df, cutoff_time = time_based_split(df)
    log.info(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows | Cutoff: {cutoff_time}")

    X_train, y_train = train_df[feature_columns], train_df["profit_margin_target"]
    X_test, y_test = test_df[feature_columns], test_df["profit_margin_target"]
    y_train_clf, y_test_clf = train_df["is_profitable"], test_df["is_profitable"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    log.info("Training regressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    log.info(f"Test MAE (profit margin, as a fraction): {mae:.4f}")

    classifier = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
    )
    log.info("Training classifier (is_profitable)...")
    classifier.fit(X_train, y_train_clf)
    classifier_proba = classifier.predict_proba(X_test)[:, 1]
    classifier_pred = (classifier_proba > PROB_THRESHOLD).astype(int)
    log.info(f"Classifier precision: {precision_score(y_test_clf, classifier_pred):.4f} | "
             f"recall: {recall_score(y_test_clf, classifier_pred):.4f} "
             f"(at prob_threshold={PROB_THRESHOLD})")

    ranking_results = evaluate_ranking(test_df, predictions, top_k=5, classifier_proba=classifier_proba)
    log.info("--- Does the model actually help pick better trades? ---")
    log.info(f"Avg actual profit margin of qualifying top-5 picks per timestamp: "
             f"{ranking_results['avg_profit_margin_of_top_picks']}")
    log.info(f"Avg profit margin across ALL products (random-pick baseline):  "
             f"{ranking_results['avg_profit_margin_random_baseline']:.4f}")
    log.info(f"Timestamps with a qualifying trade: {ranking_results['num_timestamps_with_trades']} | "
             f"Timestamps with NO trade meeting the bar: "
             f"{ranking_results['num_timestamps_with_no_qualifying_trade']}")
    log.info(f"Total trades that would have been taken: {ranking_results['total_trades_taken']}")

    backtest_results = realistic_backtest(test_df, predictions, classifier_proba=classifier_proba)
    log.info("--- Realistic backtest (capital-constrained, sequential, hybrid filter) ---")
    log.info(f"Starting capital: {backtest_results['starting_capital']:.2f} | "
             f"Final capital: {backtest_results['final_capital']:.2f} | "
             f"Return: {backtest_results['total_return_pct']:.2%}")
    log.info(f"Total trades executed: {backtest_results['total_trades']} | "
             f"Avg profit margin per trade: {backtest_results['avg_profit_margin_per_trade']}")
    log.info(f"Max drawdown: {backtest_results['max_drawdown_pct']:.2%}")

    log.info("Computing feature importance (this can take a minute)...")
    importance = permutation_importance(
        model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance.importances_mean,
    }).sort_values("importance", ascending=False)
    log.info("Top features:\n" + importance_df.head(10).to_string(index=False))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "regressor": model,
        "classifier": classifier,
        "feature_columns": feature_columns,
        "prob_threshold": PROB_THRESHOLD,
        "min_profit_threshold": MIN_PROFIT_THRESHOLD,
    }, MODEL_PATH)
    log.info(f"Saved model bundle (regressor + classifier) to {MODEL_PATH}")


if __name__ == "__main__":
    main()