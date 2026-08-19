import sqlite3
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CryptoDatabase:
    """Handles all SQLite operations for crypto price data, indicators, predictions, and AI cache."""

    def __init__(self, db_path: str = "data/crypto.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_database()

    def get_connection(self):
        """Context manager pattern for resource safety."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Create tables if they don't exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # ─── Core price data ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    price_usd REAL NOT NULL,
                    volume_24h REAL,
                    market_cap REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(coin_id, timestamp)
                )
            """)

            # ─── Fetch audit logs ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    coins_fetched INTEGER,
                    status TEXT,
                    error_message TEXT
                )
            """)

            # ─── Technical indicators (features) ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_percent REAL,
                    price_change_1h REAL,
                    price_change_24h REAL,
                    volatility REAL,
                    target INTEGER,
                    UNIQUE(coin_id, timestamp)
                )
            """)

            # ─── Prediction history ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    predicted_class INTEGER NOT NULL,
                    predicted_prob REAL NOT NULL,
                    actual_class INTEGER,
                    model_version TEXT,
                    features_hash TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ─── AI LLM cache (deduplication) ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT NOT NULL UNIQUE,
                    prompt_type TEXT NOT NULL,
                    coin_id TEXT,
                    response_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME
                )
            """)

            # ─── Drift detection logs ───
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    drift_detected INTEGER NOT NULL,
                    psi_score REAL,
                    ks_statistic REAL,
                    feature_drift TEXT,
                    narrative TEXT
                )
            """)

            conn.commit()
            logger.info("Database initialized with all tables")

    def insert_price(self, coin_id: str, price: float, volume: float = None, market_cap: float = None):
        """Insert single price record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO price_history (coin_id, price_usd, volume_24h, market_cap)
                    VALUES (?, ?, ?, ?)
                """, (coin_id, price, volume, market_cap))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate entry for {coin_id}")
                return False

    def log_fetch(self, coins_count: int, status: str, error: str = None):
        """Track every fetch operation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO fetch_logs (coins_fetched, status, error_message)
                VALUES (?, ?, ?)
            """, (coins_count, status, error))
            conn.commit()

    def get_recent_prices(self, coin_id: str, hours: int = 24) -> pd.DataFrame:
        """Get last N hours of data for model input — SQL injection fixed via parameterized query."""
        with self.get_connection() as conn:
            query = """
                SELECT timestamp, price_usd, volume_24h
                FROM price_history
                WHERE coin_id = ?
                AND timestamp >= datetime('now', ?)
                ORDER BY timestamp DESC
            """
            # hours is passed as a bound parameter, not string-interpolated
            offset_str = f"-{hours} hours"
            return pd.read_sql_query(query, conn, params=(coin_id, offset_str))

    def save_indicators(self, coin_id: str, df: pd.DataFrame):
        """Store calculated indicators in database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    coin_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    rsi REAL,
                    macd_line REAL,
                    macd_signal REAL,
                    macd_histogram REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_percent REAL,
                    price_change_1h REAL,
                    price_change_24h REAL,
                    volatility REAL,
                    target INTEGER,
                    UNIQUE(coin_id, timestamp)
                )
            """)

            valid = df.dropna()
            records = []
            for _, row in valid.iterrows():
                records.append((
                    coin_id, row['timestamp'], row['rsi'],
                    row['macd_line'], row['macd_signal'], row['macd_histogram'],
                    row['bb_upper'], row['bb_lower'], row['bb_percent'],
                    row['price_change_1h'], row['price_change_24h'],
                    row['volatility'], row['target']
                ))

            cursor.executemany("""
                INSERT OR IGNORE INTO indicators
                (coin_id, timestamp, rsi, macd_line, macd_signal, macd_histogram,
                 bb_upper, bb_lower, bb_percent, price_change_1h, price_change_24h,
                 volatility, target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)

            conn.commit()
            logger.info(f"Inserted {cursor.rowcount} indicator records")

    # ─────────────────────────────────────────────
    # Prediction History
    # ─────────────────────────────────────────────

    def log_prediction(
        self,
        coin_id: str,
        predicted_class: int,
        predicted_prob: float,
        model_version: Optional[str] = None,
        features_hash: Optional[str] = None,
        actual_class: Optional[int] = None,
    ) -> int:
        """Log a model prediction for backtesting and auditing."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prediction_history
                (coin_id, predicted_class, predicted_prob, model_version, features_hash, actual_class)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (coin_id, predicted_class, predicted_prob, model_version, features_hash, actual_class))
            conn.commit()
            return cursor.lastrowid

    def get_predictions(
        self,
        coin_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Retrieve prediction history for analysis."""
        with self.get_connection() as conn:
            query = "SELECT * FROM prediction_history WHERE 1=1"
            params: List[Any] = []
            if coin_id:
                query += " AND coin_id = ?"
                params.append(coin_id)
            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            return pd.read_sql_query(query, conn, params=params)

    def update_prediction_actual(self, prediction_id: int, actual_class: int):
        """Backfill the actual outcome after the fact."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prediction_history
                SET actual_class = ?
                WHERE id = ?
            """, (actual_class, prediction_id))
            conn.commit()

    # ─────────────────────────────────────────────
    # AI Cache
    # ─────────────────────────────────────────────

    def get_ai_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached LLM response if not expired."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT response_json, expires_at
                FROM ai_cache
                WHERE cache_key = ?
                AND (expires_at IS NULL OR expires_at > datetime('now'))
            """, (cache_key,))
            row = cursor.fetchone()
            if row:
                import json
                return json.loads(row[0])
            return None

    def set_ai_cache(
        self,
        cache_key: str,
        prompt_type: str,
        response_json: str,
        coin_id: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ):
        """Cache an LLM response to avoid redundant calls."""
        expires = None
        if ttl_hours:
            expires = datetime.utcnow().replace(microsecond=0).isoformat()
            # SQLite will compare as text; for strict correctness we rely on client logic
            # but here we store ISO string and check in get_ai_cache
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_cache (cache_key, prompt_type, coin_id, response_json, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = datetime('now'),
                    expires_at = excluded.expires_at
            """, (cache_key, prompt_type, coin_id, response_json, expires))
            conn.commit()

    def prune_ai_cache(self, max_age_days: int = 7):
        """Remove stale cache entries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM ai_cache
                WHERE created_at < datetime('now', ?)
            """, (f"-{max_age_days} days",))
            conn.commit()
            logger.info(f"Pruned {cursor.rowcount} stale AI cache entries")

    # ─────────────────────────────────────────────
    # Drift Logs
    # ─────────────────────────────────────────────

    def log_drift(
        self,
        coin_id: str,
        drift_detected: bool,
        psi_score: Optional[float] = None,
        ks_statistic: Optional[float] = None,
        feature_drift: Optional[str] = None,
        narrative: Optional[str] = None,
    ):
        """Record a drift detection event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drift_logs
                (coin_id, drift_detected, psi_score, ks_statistic, feature_drift, narrative)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (coin_id, int(drift_detected), psi_score, ks_statistic, feature_drift, narrative))
            conn.commit()

    def get_recent_drift(self, coin_id: str, limit: int = 10) -> pd.DataFrame:
        """Get recent drift detection results."""
        with self.get_connection() as conn:
            return pd.read_sql_query("""
                SELECT * FROM drift_logs
                WHERE coin_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=(coin_id, limit))

    # ─────────────────────────────────────────────
    # Utility
    # ─────────────────────────────────────────────

    def get_indicator_window(
        self,
        coin_id: str,
        lookback: int = 100,
    ) -> pd.DataFrame:
        """Fetch the most recent indicator rows for LLM context."""
        with self.get_connection() as conn:
            return pd.read_sql_query("""
                SELECT * FROM indicators
                WHERE coin_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, conn, params=(coin_id, lookback))