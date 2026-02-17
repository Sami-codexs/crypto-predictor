import sqlite3
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CryptoDatabase:
    """Handles all SQLite operations for crypto price data."""
    
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
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fetch_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    coins_fetched INTEGER,
                    status TEXT,
                    error_message TEXT
                )
            """)
            
            conn.commit()
            logger.info("Database initialized")
    
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
    
    def get_recent_prices(self, coin_id: str, hours: int = 24):
        """Get last N hours of data for model input."""
        with self.get_connection() as conn:
            query = """
                SELECT timestamp, price_usd, volume_24h 
                FROM price_history 
                WHERE coin_id = ? 
                AND timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp DESC
            """.format(hours)
            return pd.read_sql_query(query, conn, params=(coin_id,))
    
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
                    macd_hist REAL,
                    bb_upper REAL,
                    bb_lower REAL,
                    bb_percent REAL,
                    price_change_1h REAL,
                    price_change_24h REAL,
                    volatility_24h REAL,
                    target INTEGER,
                    UNIQUE(coin_id, timestamp)
                )
            """)
            
            # Drop rows with NaN values (warmup period)
            valid = df.dropna()
            
            records = []
            for _, row in valid.iterrows():
                records.append((
                    coin_id, row['timestamp'], row['rsi'],
                    row['macd_line'], row['macd_signal'], row['macd_hist'],
                    row['bb_upper'], row['bb_lower'], row['bb_percent'],
                    row['price_change_1h'], row['price_change_24h'],
                    row['volatility_24h'], row['target']
                ))
            
            cursor.executemany("""
                INSERT OR IGNORE INTO indicators 
                (coin_id, timestamp, rsi, macd_line, macd_signal, macd_hist,
                 bb_upper, bb_lower, bb_percent, price_change_1h, price_change_24h,
                 volatility_24h, target)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            conn.commit()
            logger.info(f"Inserted {cursor.rowcount} indicator records")