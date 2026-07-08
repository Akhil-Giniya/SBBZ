"""
Hypixel Skyblock Bazaar data collector

Architecture:
- SQLite WAL buffering (Crash-proof, zero data loss).
- Idempotent Daily Parquet Compaction (Exactly-once delivery, atomic renames).
- HTTP ETags + urllib3 Connection Pooling & Auto-Retries.
- Robust JSON parsing (Gracefully handles unexpected API schema changes).
- Disk space monitoring with automatic safe shutdown.
"""

import os
import time
import signal
import logging
from logging.handlers import RotatingFileHandler
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------- Config ----------------
API_URL = "https://api.hypixel.net/v2/skyblock/bazaar"
POLL_INTERVAL = 60
SHUTDOWN_CHECK_INTERVAL = 1
DATA_DIR = Path("data/skyblock_bazaar")
DB_FILE = DATA_DIR / "bazaar_buffer.db"
LOG_FILE = "collector.log"
REQUEST_TIMEOUT = 15

DISK_CHECK_INTERVAL = 300
WARN_FREE_DISK_MB = 1000
CRITICAL_FREE_DISK_MB = 200
HARD_STOP_FREE_DISK_MB = 100  # Triggers safe shutdown to prevent OS lockup

DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ],
)
log = logging.getLogger("bazaar_collector")

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

# ---------------- Database & Storage ----------------

def init_db(db_path: Path):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row

    # Pragmas for resilience and auto-maintenance
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA auto_vacuum=INCREMENTAL;")

    # Metadata table for future-proofing schema changes
    conn.execute('''
        CREATE TABLE IF NOT EXISTS _metadata (
            schema_version INTEGER,
            collector_version INTEGER,
            created_at TEXT
        )
    ''')

    # Initialize metadata if empty
    if conn.execute("SELECT COUNT(*) FROM _metadata").fetchone()[0] == 0:
        conn.execute("INSERT INTO _metadata VALUES (1, 1, ?)", (datetime.now(timezone.utc).isoformat(),))

    conn.execute('''
        CREATE TABLE IF NOT EXISTS buffer (
            timestamp TEXT, product TEXT, buy_price REAL, sell_price REAL,
            buy_volume INTEGER, sell_volume INTEGER, buy_orders INTEGER,
            sell_orders INTEGER, buy_moving_week REAL, sell_moving_week REAL,
            spread REAL, mid_price REAL, demand_supply_ratio REAL,
            hour INTEGER, day_of_week INTEGER
        )
    ''')

    # Indexes for fast querying and fast midnight compaction
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON buffer(timestamp);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_buffer_date ON buffer(date(timestamp));")
    conn.commit()
    return conn

def insert_rows_to_db(conn, rows):
    conn.executemany('''
        INSERT INTO buffer VALUES (
            :timestamp, :product, :buy_price, :sell_price,
            :buy_volume, :sell_volume, :buy_orders, :sell_orders,
            :buy_moving_week, :sell_moving_week, :spread,
            :mid_price, :demand_supply_ratio, :hour, :day_of_week
        )
    ''', rows)
    conn.commit()

def flush_completed_days_to_parquet(conn, current_utc_day: str):
    cursor = conn.execute(
        "SELECT DISTINCT date(timestamp) FROM buffer WHERE date(timestamp) < ?",
        (current_utc_day,)
    )
    completed_days = [row[0] for row in cursor.fetchall()]

    for day_str in completed_days:
        log.info(f"Compacting data for completed day: {day_str}...")

        cursor = conn.execute("SELECT * FROM buffer WHERE date(timestamp) = ?", (day_str,))
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            continue

        for r in rows:
            r['timestamp'] = datetime.fromisoformat(r['timestamp'])

        day_dir = DATA_DIR / day_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # Idempotent filenames (no part2/part3 loops)
        final_path = day_dir / "daily_bazaar.parquet"
        temp_path = day_dir / "daily_bazaar.parquet.tmp"

        # A day can be compacted more than once: a row's date comes from
        # the API's `lastUpdated`, not wall-clock time, so a poll just
        # after midnight can still land in "yesterday" if the API's
        # snapshot lagged. That stray row gets inserted into a day that
        # was already compacted and cleared from the buffer, and won't be
        # picked up again until the next day-rollover. If we only wrote
        # what's currently in the buffer, this second pass would contain
        # just the stray row(s) and would overwrite -- and destroy -- the
        # already-complete file from the first pass. Merge with whatever
        # is already on disk instead, so a re-run only ever adds rows.
        combined = {}
        if final_path.exists():
            try:
                for r in pq.read_table(final_path).to_pylist():
                    combined[(r["timestamp"], r["product"])] = r
            except Exception as e:
                log.error(
                    f"Could not read existing {final_path} to merge; skipping "
                    f"compaction for {day_str} this round to avoid overwriting it: {e}"
                )
                continue

        for r in rows:
            combined[(r["timestamp"], r["product"])] = r
        merged_rows = list(combined.values())

        table = pa.Table.from_pylist(merged_rows, schema=SCHEMA)

        try:
            pq.write_table(table, temp_path, compression="snappy")

            # Force OS to flush the file buffers to disk hardware
            with open(temp_path, "rb") as f:
                os.fsync(f.fileno())

            # Atomic rename guarantees exactly-once delivery.
            temp_path.replace(final_path)

            # Also fsync the containing directory so the rename itself
            # is durable across a power loss, not just a process crash --
            # a bare file fsync only guarantees the *content* landed.
            dir_fd = os.open(day_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

            log.info(f"Wrote {len(merged_rows)} rows to {final_path} ({len(rows)} from this pass)")

        except Exception as e:
            log.error(f"Failed to write parquet for {day_str}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            continue # Skip SQLite deletion to prevent data loss

        # Safely delete flushed rows and shrink DB
        conn.execute("DELETE FROM buffer WHERE date(timestamp) = ?", (day_str,))
        conn.execute("PRAGMA incremental_vacuum;")
        conn.commit()


# ---------------- API & Utility Functions ----------------

def check_disk_space(path: Path):
    usage = shutil.disk_usage(path)
    free_mb = usage.free / (1024 * 1024)

    if free_mb < HARD_STOP_FREE_DISK_MB:
        log.critical(f"FATAL: Disk space < {HARD_STOP_FREE_DISK_MB}MB. Forcing clean shutdown.")
        global _shutdown_requested
        _shutdown_requested = True
    elif free_mb < CRITICAL_FREE_DISK_MB:
        log.critical(f"Disk space critically low: {free_mb:.0f} MB free at {path}.")
    elif free_mb < WARN_FREE_DISK_MB:
        log.warning(f"Disk space getting low: {free_mb:.0f} MB free at {path}.")

    return free_mb

def setup_session() -> requests.Session:
    session = requests.Session()
    # Automated retries for network drops and 500-level server errors
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def fetch_bazaar(session: requests.Session, etag: str = None):
    headers = {"If-None-Match": etag} if etag else {}

    try:
        resp = session.get(API_URL, headers=headers, timeout=REQUEST_TIMEOUT)

        if resp.status_code == 304:
            return None, etag

        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", True) or "products" not in data:
            raise ValueError(f"API returned invalid payload: {data}")

        return data, resp.headers.get("ETag")

    except Exception as e:
        log.warning(f"Fetch failed (urllib3 handled retries): {e}. Skipping this poll.")
        return None, etag

def parse_bazaar(data, timestamp: datetime):
    rows = []
    ts_str = timestamp.isoformat()

    for name, item in data.get("products", {}).items():
        qs = item.get("quick_status", {})

        # Robust parsing: .get() maps missing API fields to None (null in Parquet)
        buy_price = qs.get("buyPrice")
        sell_price = qs.get("sellPrice")
        buy_volume = qs.get("buyVolume", 0)
        sell_volume = qs.get("sellVolume", 0)

        spread = None
        mid_price = None
        if buy_price is not None and sell_price is not None:
            spread = sell_price - buy_price
            mid_price = (sell_price + buy_price) / 2

        ds_ratio = None
        if buy_volume is not None and sell_volume is not None:
            ds_ratio = buy_volume / (sell_volume + 1)

        rows.append({
            "timestamp": ts_str,
            "product": name,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_orders": qs.get("buyOrders"),
            "sell_orders": qs.get("sellOrders"),
            "buy_moving_week": qs.get("buyMovingWeek"),
            "sell_moving_week": qs.get("sellMovingWeek"),
            "spread": spread,
            "mid_price": mid_price,
            "demand_supply_ratio": ds_ratio,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
        })
    return rows


# ---------------- Main Loop & Shutdown ----------------

_shutdown_requested = False

def _request_shutdown(signum, frame):
    global _shutdown_requested
    log.info(f"Received signal {signum}, requesting shutdown...")
    _shutdown_requested = True

def sleep_with_shutdown_check(total_seconds: float):
    end = time.time() + max(0, total_seconds)
    while time.time() < end and not _shutdown_requested:
        time.sleep(min(SHUTDOWN_CHECK_INTERVAL, max(0, end - time.time())))

def main():
    db_conn = init_db(DB_FILE)

    try:
        session = setup_session()

        last_etag = None
        last_disk_check = 0.0

        # Seed dedup state from whatever's already buffered, so a restart
        # right after a poll doesn't forget it and insert a duplicate row
        # for data it already captured moments before the crash.
        last_update_ms = None
        row = db_conn.execute("SELECT MAX(timestamp) AS ts FROM buffer").fetchone()
        if row and row["ts"]:
            last_update_ms = int(datetime.fromisoformat(row["ts"]).timestamp() * 1000)
        current_utc_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        signal.signal(signal.SIGINT, _request_shutdown)
        signal.signal(signal.SIGTERM, _request_shutdown)

        log.info("Starting enterprise bazaar collector...")

        # Boot check: Process any data from previous days left in the buffer
        flush_completed_days_to_parquet(db_conn, current_utc_day)

        while not _shutdown_requested:
            loop_start = time.time()

            if loop_start - last_disk_check >= DISK_CHECK_INTERVAL:
                check_disk_space(DATA_DIR)
                last_disk_check = loop_start

            if _shutdown_requested:
                break

            data, last_etag = fetch_bazaar(session, last_etag)

            if data is not None:
                update_ms = data.get("lastUpdated")
                if update_ms is not None:
                    timestamp = datetime.fromtimestamp(update_ms / 1000, tz=timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)
                    update_ms = None

                if update_ms is not None and update_ms == last_update_ms:
                    log.info("API data unchanged (stale lastUpdated); skipping.")
                else:
                    rows = parse_bazaar(data, timestamp)
                    if rows:
                        insert_rows_to_db(db_conn, rows)
                        last_update_ms = update_ms
                        log.info(f"Collected and buffered {len(rows)} products at {timestamp.isoformat()}")

            now_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if now_day != current_utc_day:
                log.info(f"UTC day crossed ({current_utc_day} -> {now_day}). Initiating Parquet compaction...")
                flush_completed_days_to_parquet(db_conn, now_day)
                current_utc_day = now_day

            elapsed = time.time() - loop_start
            sleep_with_shutdown_check(POLL_INTERVAL - elapsed)

        log.info("Clean shutdown initiated. Unwritten daily data remains safely in SQLite buffer.")

    finally:
        # Ensures the database connection is cleanly closed even if the script crashes
        db_conn.close()
        log.info("Database connection closed. Exiting.")

    return

if __name__ == "__main__":
    main()