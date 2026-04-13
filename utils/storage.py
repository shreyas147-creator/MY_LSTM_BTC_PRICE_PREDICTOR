"""
utils/storage.py — Parquet read/write and SQLite helpers.
All data persistence goes through here — never call pd.to_parquet directly.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List
from utils.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, path: Path, verbose: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=True, compression="snappy")
    if verbose:
        logger.info(f"Saved {len(df):,} rows → {path.name}  ({path.stat().st_size / 1024:.1f} KB)")


def load_parquet(path: Path, verbose: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    if verbose:
        logger.info(f"Loaded {len(df):,} rows ← {path.name}")
    return df

def append_parquet(df: pd.DataFrame, path: Path, verbose: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = load_parquet(path, verbose=False)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
        except Exception:
            combined = df
    else:
        combined = df
    save_parquet(combined, path, verbose=verbose)
    return combined


def parquet_exists(path: Path) -> bool:
    return Path(path).exists()


# ---------------------------------------------------------------------------
# SQLite — always use context manager so connection is closed on Windows
# ---------------------------------------------------------------------------

@contextmanager
def sqlite_connection(db_path: Path):
    """
    Context manager that opens, yields, commits, and CLOSES the connection.
    Closing is mandatory on Windows to release the file lock.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()       # ← this is the fix: explicit close releases WinError 32


def sqlite_execute(
    db_path: Path,
    sql: str,
    params: tuple = (),
) -> None:
    """Execute a single statement (CREATE, INSERT, UPDATE, DELETE)."""
    with sqlite_connection(db_path) as conn:
        conn.execute(sql, params)


def sqlite_query(
    db_path: Path,
    sql: str,
    params: tuple = (),
) -> pd.DataFrame:
    """Run a SELECT and return a DataFrame."""
    with sqlite_connection(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def sqlite_insert_df(
    df: pd.DataFrame,
    db_path: Path,
    table: str,
    if_exists: str = "append",
) -> None:
    """Write a DataFrame to a SQLite table."""
    with sqlite_connection(db_path) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=True)
    logger.info(f"SQLite: wrote {len(df):,} rows → {table} @ {db_path.name}")


def sqlite_table_exists(db_path: Path, table: str) -> bool:
    """Check if a table exists in the database."""
    with sqlite_connection(db_path) as conn:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return cur.fetchone() is not None


def sqlite_last_timestamp(
    db_path: Path,
    table: str,
    ts_col: str = "timestamp",
) -> Optional[str]:
    """Return the most recent timestamp in a table, or None if empty."""
    if not db_path.exists() or not sqlite_table_exists(db_path, table):
        return None
    with sqlite_connection(db_path) as conn:
        cur = conn.execute(f"SELECT MAX({ts_col}) FROM {table}")
        row = cur.fetchone()
        return row[0] if row else None

class SQLiteStore:
    """Thin wrapper with key-value store + fetch log."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _init_tables(self):
        with sqlite_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    source    TEXT,
                    timeframe TEXT,
                    last_ts   TEXT,
                    n_rows    INTEGER,
                    fetched_at TEXT DEFAULT (datetime('now')),
                    PRIMARY KEY (source, timeframe)
                )
            """)

    def set(self, key: str, value: str) -> None:
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)",
                (key, str(value))
            )

    def get(self, key: str, default: str = None) -> Optional[str]:
        with sqlite_connection(self.db_path) as conn:
            cur = conn.execute("SELECT value FROM kv_store WHERE key=?", (key,))
            row = cur.fetchone()
        return row[0] if row else default

    def log_fetch(self, source: str, timeframe: str,
                  last_ts: str, n_rows: int) -> None:
        with sqlite_connection(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO fetch_log (source, timeframe, last_ts, n_rows)
                VALUES (?, ?, ?, ?)
            """, (source, timeframe, str(last_ts), int(n_rows)))

    def last_fetch_ts(self, source: str, timeframe: str) -> Optional[str]:
        with sqlite_connection(self.db_path) as conn:
            cur = conn.execute(
                "SELECT last_ts FROM fetch_log WHERE source=? AND timeframe=?",
                (source, timeframe)
            )
            row = cur.fetchone()
        return row[0] if row else None

    def execute(self, sql: str, params: tuple = ()) -> None:
        sqlite_execute(self.db_path, sql, params)

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return sqlite_query(self.db_path, sql, params)

    def insert_df(self, df: pd.DataFrame, table: str,
                  if_exists: str = "append") -> None:
        sqlite_insert_df(df, self.db_path, table, if_exists)

    def table_exists(self, table: str) -> bool:
        return sqlite_table_exists(self.db_path, table)

    def last_timestamp(self, table: str, ts_col: str = "timestamp"):
        return sqlite_last_timestamp(self.db_path, table, ts_col)