"""
Hypixel Skyblock Bazaar data collector.

Polls the bazaar API every POLL_INTERVAL seconds and writes the data to
daily-rotated Parquet files under DATA_DIR. Designed to run continuously
for weeks/months: buffers rows in memory, flushes to disk periodically,
retries on network failures, rotates to a new file each UTC day, and is
safe against overwriting data if the script restarts mid-day.

Usage:
    pip install requests pandas pyarrow
    python bazaar_collector.py

Stop safely with Ctrl+C (SIGINT) - it flushes the buffer before exiting.

Loading the data later for ML:
    import pandas as pd
    df = pd.read_parquet("data/skyblock_bazaar/")   # reads every file in the folder
"""

import time
import sys
import signal
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------- Config ----------------
API_URL = "https://api.hypixel.net/v2/skyblock/bazaar"
POLL_INTERVAL = 60        # seconds between API polls
FLUSH_INTERVAL = 300      # seconds between disk writes (buffers ~5 polls)
DATA_DIR = Path("data/skyblock_bazaar")
LOG_FILE = "collector.log"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 5
MAX_BUFFER_SIZE = 100_000  # hard cap: force-flush (or drop oldest) if flushing is failing

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("bazaar_collector")

# Fixed, explicit schema. This is what actually prevents schema-mismatch
# crashes -- we always build tables against this, never infer it from
# whatever a given batch of data looks like.
SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("product", pa.string()),
    ("buy_price", pa.float64()),
    ("sell_price", pa.float64()),
    ("buy_volume", pa.int64()),
    ("sell_volume", pa.int64()),
    ("buy_orders", pa.int64()),
    ("sell_orders", pa.int64()),
    ("buy_moving_week", pa.float64()),
    ("sell_moving_week", pa.float64()),
    ("spread", pa.float64()),
    ("mid_price", pa.float64()),
    ("demand_supply_ratio", pa.float64()),
    ("hour", pa.int64()),
    ("day_of_week", pa.int64()),
])


class ParquetDayWriter:
    """Writes buffered rows to a Parquet file, rotating to a new file each UTC day.

    If a file for the current day already exists on disk (e.g. because the
    script restarted), it writes to a new '_partN' file instead of
    overwriting the existing one. All files in the directory are read
    together at load time, so this is invisible to downstream code.
    """

    def __init__(self, data_dir: Path, schema: pa.Schema):
        self.data_dir = data_dir
        self.schema = schema
        self.writer = None
        self.current_date = None
        self.current_path = None

    def _pick_path(self, date_str: str) -> Path:
        base = self.data_dir / f"{date_str}.parquet"
        if not base.exists():
            return base
        i = 2
        while True:
            candidate = self.data_dir / f"{date_str}_part{i}.parquet"
            if not candidate.exists():
                return candidate
            i += 1

    def write(self, df: pd.DataFrame):
        if df.empty:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if date_str != self.current_date:
            self._close()
            self.current_date = date_str
            self.current_path = self._pick_path(date_str)
            self.writer = pq.ParquetWriter(
                self.current_path, self.schema, compression="snappy"
            )
            log.info(f"Opened new file: {self.current_path.name}")

        table = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self.writer.write_table(table)

    def _close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def close(self):
        self._close()


def fetch_bazaar():
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("success", True):
                raise ValueError(f"API returned success=false: {data}")
            if "products" not in data:
                raise ValueError(f"API response missing 'products' key: {data}")
            return data
        except Exception as e:
            wait = min(2 ** attempt, 60)
            log.warning(f"Fetch attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s")
            time.sleep(wait)
    log.error("All retry attempts failed; skipping this poll.")
    return None


def parse_bazaar(data, timestamp):
    rows = []
    for name, item in data["products"].items():
        qs = item["quick_status"]
        buy_price = qs["buyPrice"]
        sell_price = qs["sellPrice"]
        buy_volume = qs["buyVolume"]
        sell_volume = qs["sellVolume"]
        rows.append({
            "timestamp": timestamp,
            "product": name,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_orders": qs["buyOrders"],
            "sell_orders": qs["sellOrders"],
            "buy_moving_week": qs.get("buyMovingWeek"),
            "sell_moving_week": qs.get("sellMovingWeek"),
            # derived features
            "spread": sell_price - buy_price,
            "mid_price": (sell_price + buy_price) / 2,
            "demand_supply_ratio": buy_volume / (sell_volume + 1),
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
        })
    return rows


def main():
    writer = ParquetDayWriter(DATA_DIR, SCHEMA)
    buffer = []
    last_flush = time.time()

    def handle_shutdown(signum, frame):
        log.info("Shutdown signal received, flushing buffer...")
        if buffer:
            try:
                writer.write(pd.DataFrame(buffer))
            except Exception as e:
                log.error(f"Failed to flush buffer on shutdown, data may be lost: {e}")
        writer.close()
        log.info("Clean shutdown complete.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    log.info("Starting bazaar collector...")

    while True:
        loop_start = time.time()
        timestamp = datetime.now(timezone.utc)

        data = fetch_bazaar()
        if data is not None:
            rows = parse_bazaar(data, timestamp)
            buffer.extend(rows)
            log.info(f"Collected {len(rows)} products at {timestamp.isoformat()} (buffer={len(buffer)})")

        should_flush = (time.time() - last_flush >= FLUSH_INTERVAL) or (len(buffer) >= MAX_BUFFER_SIZE)

        if should_flush and buffer:
            try:
                df = pd.DataFrame(buffer)
                writer.write(df)
                log.info(f"Flushed {len(buffer)} rows to {writer.current_path.name}")
                buffer = []
                last_flush = time.time()
            except Exception as e:
                log.error(f"Flush failed, will retry next cycle: {e}")
                # Don't reset last_flush -- we'll try again next loop.
                # But if the buffer keeps growing past the hard cap because
                # disk writes are failing, drop the oldest rows rather than
                # let memory grow unbounded for a 6-month run.
                if len(buffer) >= MAX_BUFFER_SIZE:
                    dropped = len(buffer) - MAX_BUFFER_SIZE // 2
                    log.error(f"Buffer still over cap after failed flush; dropping oldest {dropped} rows.")
                    buffer = buffer[-MAX_BUFFER_SIZE // 2:]

        elapsed = time.time() - loop_start
        time.sleep(max(0, POLL_INTERVAL - elapsed))


if __name__ == "__main__":
    main()