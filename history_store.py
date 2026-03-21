"""
history_store.py — ApexAlgo Historical OHLCV Cache
======================================================
Fetches candle data from yfinance and caches it in a local SQLite database.
Supports all trading symbols and timeframes used by the bot.

Designed to:
  - Pre-warm strategy indicators before the first live trade (no cold-start)
  - Provide candle data for the dashboard/mobile charts without MT5
  - Allow demo mode to generate real signals when MT5 is not connected
  - Work fully offline after the initial data download

Usage:
    from history_store import HistoryStore
    hs = HistoryStore()
    df = hs.get_candles("XAUUSD", "H1", 200)   # fast: reads from cache
    hs.refresh_all()                             # update all symbols/timeframes
"""

import os
import sqlite3
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd  # type: ignore

logger = logging.getLogger("apexalgo.history")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Map internal symbol names → yfinance tickers
SYMBOL_MAP: dict[str, str] = {
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD",
}

# Map internal timeframe → (yfinance interval, yfinance period for initial load)
TIMEFRAME_MAP: dict[str, tuple[str, str]] = {
    "M5":  ("5m",  "60d"),
    "M15": ("15m", "60d"),
    "H1":  ("1h",  "730d"),   # 2 years on H1
    "H4":  ("1d",  "730d"),   # yfinance has no native 4H; we resample from 1D
    "D1":  ("1d",  "1825d"),  # 5 years on daily
}

# Minimum age before we consider data stale and re-download
REFRESH_THRESHOLD: dict[str, int] = {
    "M5":  5,     # minutes
    "M15": 15,
    "H1":  60,
    "H4":  240,
    "D1":  1440,  # 1 day in minutes
}

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "history.db")


# ──────────────────────────────────────────────────────────────
# HistoryStore
# ──────────────────────────────────────────────────────────────

class HistoryStore:
    """
    SQLite-backed OHLCV candle cache.
    Thread-safe — all writes are serialised via an internal lock.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._lock   = threading.Lock()
        self._ensure_db()
        logger.info(f"[History] Store initialised at {self.db_path}")

    # ── Initialisation ────────────────────────────────────────

    def _ensure_db(self):
        """Create the data directory and candles table if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    symbol      TEXT    NOT NULL,
                    timeframe   TEXT    NOT NULL,
                    ts          INTEGER NOT NULL,   -- unix timestamp (seconds)
                    open        REAL    NOT NULL,
                    high        REAL    NOT NULL,
                    low         REAL    NOT NULL,
                    close       REAL    NOT NULL,
                    volume      REAL    NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, timeframe, ts)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    symbol      TEXT    NOT NULL,
                    timeframe   TEXT    NOT NULL,
                    last_update INTEGER NOT NULL,
                    row_count   INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, timeframe)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_candles_sym_tf_ts "
                "ON candles (symbol, timeframe, ts DESC)"
            )
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Staleness Check ───────────────────────────────────────

    def _last_update(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Return last cache update time for a symbol/timeframe pair, or None."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_update FROM cache_meta WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
        if row is None:
            return None
        return datetime.fromtimestamp(row["last_update"], tz=timezone.utc)

    def _is_stale(self, symbol: str, timeframe: str) -> bool:
        """Return True if the cache is missing or older than REFRESH_THRESHOLD."""
        last = self._last_update(symbol, timeframe)
        if last is None:
            return True
        threshold_minutes = REFRESH_THRESHOLD.get(timeframe, 60)
        return datetime.now(timezone.utc) - last > timedelta(minutes=threshold_minutes)

    # ── Download & Cache ──────────────────────────────────────

    def fetch_and_cache(self, symbol: str, timeframe: str,
                        days: Optional[int] = None,
                        force: bool = False) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance and store it in SQLite.
        Returns the downloaded DataFrame (may be empty on network error).
        Skips download if cache is fresh enough (unless force=True).
        """
        if not force and not self._is_stale(symbol, timeframe):
            logger.debug(f"[History] Cache fresh: {symbol} {timeframe} — skipping download")
            return self.get_candles(symbol, timeframe)

        yf_ticker = SYMBOL_MAP.get(symbol)
        if yf_ticker is None:
            logger.warning(f"[History] Unknown symbol: {symbol}")
            return pd.DataFrame()

        yf_interval, default_period = TIMEFRAME_MAP.get(timeframe, ("1h", "730d"))
        period = f"{days}d" if days else default_period

        # H4 is not natively supported by yfinance — download 1H and resample
        resample_4h = (timeframe == "H4")
        if resample_4h:
            yf_interval = "1h"

        try:
            import yfinance as yf  # type: ignore
            logger.info(
                f"[History] Downloading {symbol} ({yf_ticker}) "
                f"interval={yf_interval} period={period}"
            )
            
            ticker = yf.Ticker(yf_ticker)
            df = ticker.history(
                period=period,
                interval=yf_interval,
                proxy=None,
                prepost=False,
                actions=False,
                auto_adjust=True,
                back_adjust=False,
            )
            
            if df.empty:  # type: ignore
                logger.warning(f"[History] yfinance returned empty data for {symbol}")
                return df  # type: ignore

            # Normalise columns
            df.columns = [  # type: ignore
                c[0].lower() if isinstance(c, tuple) else c.lower()
                for c in df.columns  # type: ignore
            ]
            if "adj close" in df.columns:  # type: ignore
                df = df.rename(columns={"adj close": "close"})  # type: ignore
            df = df[["open", "high", "low", "close", "volume"]].dropna()  # type: ignore

            # Resample 1H → 4H if needed
            if resample_4h:
                df = df.resample("4h").agg({
                    "open":   "first",
                    "high":   "max",
                    "low":    "min",
                    "close":  "last",
                    "volume": "sum",
                }).dropna()

            self._write_to_db(symbol, timeframe, df)
            logger.info(
                f"[History] Cached {len(df)} candles for {symbol} {timeframe}"
            )
            return df

        except Exception as e:
            logger.error(f"[History] Download error for {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def _write_to_db(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """Upsert OHLCV rows into SQLite."""
        if df.empty:
            return
        rows: list[tuple] = []
        for ts, row in df.iterrows():
            # Ensure we always have a pd.Timestamp regardless of index type
            unix_ts: int = int(pd.Timestamp(ts).timestamp())  # type: ignore
            rows.append((
                symbol, timeframe, unix_ts,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row.get("volume", 0)),
            ))

        with self._lock:
            with self._get_conn() as conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO candles
                       (symbol, timeframe, ts, open, high, low, close, volume)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    rows,
                )
                conn.execute(
                    """INSERT OR REPLACE INTO cache_meta
                       (symbol, timeframe, last_update, row_count)
                       VALUES (?, ?, ?, ?)""",
                    (symbol, timeframe, int(datetime.now(timezone.utc).timestamp()), len(rows)),
                )
                conn.commit()

    # ── Read ──────────────────────────────────────────────────

    def get_candles(self, symbol: str, timeframe: str,
                    limit: int = 500) -> pd.DataFrame:
        """
        Return the most recent `limit` candles from the SQLite cache.
        Returns an empty DataFrame if no data is stored yet.
        Column order: open, high, low, close, volume  — indexed by datetime.
        """
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT ts, open, high, low, close, volume
                   FROM candles
                   WHERE symbol=? AND timeframe=?
                   ORDER BY ts DESC
                   LIMIT ?""",
                (symbol, timeframe, limit),
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [dict(r) for r in rows],
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        df = df.set_index("time").drop(columns=["ts"])
        df = df.sort_index()                          # ascending time order
        return df

    def get_candles_json(self, symbol: str, timeframe: str,
                         limit: int = 200) -> list[dict]:
        """
        Return candles as a list of dicts suitable for JSON serialisation.
        Each dict has: time (ISO 8601), open, high, low, close, volume.
        """
        df = self.get_candles(symbol, timeframe, limit)
        if df.empty:
            return []
        records: list[dict] = []
        for ts, row in df.iterrows():
            o, h, l, c, v = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row.get("volume", 0))
            records.append({
                "time":   pd.Timestamp(ts).isoformat(),  # type: ignore
                "open":   round(o, 5),
                "high":   round(h, 5),
                "low":    round(l, 5),
                "close":  round(c, 5),
                "volume": round(v, 2),
            })
        return records

    def get_last_close(self, symbol: str, timeframe: str = "H1") -> Optional[float]:
        """Return the most recent close price for a symbol, or None if no cache."""
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT close FROM candles
                   WHERE symbol=? AND timeframe=?
                   ORDER BY ts DESC LIMIT 1""",
                (symbol, timeframe),
            ).fetchone()
        return float(row["close"]) if row else None

    # ── Cache Metadata ────────────────────────────────────────

    def cache_info(self, symbol: str, timeframe: str) -> dict:
        """Return metadata about the cached data for a symbol/timeframe."""
        with self._get_conn() as conn:
            meta = conn.execute(
                "SELECT last_update, row_count FROM cache_meta WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            ).fetchone()
            if meta is None:
                return {"symbol": symbol, "timeframe": timeframe,
                        "cached": False, "row_count": 0, "last_update": None}

            # Min/max price and average volume from stored rows
            stats = conn.execute(
                """SELECT MIN(low) as price_min, MAX(high) as price_max,
                          AVG(volume) as avg_volume
                   FROM candles WHERE symbol=? AND timeframe=?""",
                (symbol, timeframe),
            ).fetchone()

        last_dt = datetime.fromtimestamp(meta["last_update"], tz=timezone.utc)
        p_min = float(stats["price_min"] or 0)
        p_max = float(stats["price_max"] or 0)
        a_vol = float(stats["avg_volume"] or 0)

        return {
            "symbol":      symbol,
            "timeframe":   timeframe,
            "cached":      True,
            "row_count":   meta["row_count"],
            "last_update": last_dt.isoformat(),
            "stale":       self._is_stale(symbol, timeframe),
            "price_min":   round(p_min, 5),
            "price_max":   round(p_max, 5),
            "avg_volume":  round(a_vol, 2),
        }

    # ── Bulk Refresh ──────────────────────────────────────────

    def refresh_all(self, symbols: Optional[list[str]] = None,
                    timeframes: Optional[list[str]] = None):
        """
        Download and cache all symbol × timeframe combinations.
        Skips any that are still fresh. Called on bot startup.
        """
        syms = symbols or list(SYMBOL_MAP.keys())
        tfs  = timeframes or list(TIMEFRAME_MAP.keys())
        logger.info(f"[History] Refreshing history: {syms} × {tfs}")
        for sym in syms:
            for tf in tfs:
                self.fetch_and_cache(sym, tf)

    def refresh_all_background(self, symbols: Optional[list[str]] = None,
                               timeframes: Optional[list[str]] = None):
        """Non-blocking version of refresh_all — runs in a daemon thread."""
        t = threading.Thread(
            target=self.refresh_all,
            kwargs={"symbols": symbols, "timeframes": timeframes},
            daemon=True,
            name="history-refresh",
        )
        t.start()
        return t
