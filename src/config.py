import logging
import sys
import os
from pathlib import Path
from typing import Optional
import os
import streamlit as st

def get_secret(key: str, default=None):
    """Read secret from Streamlit Cloud secrets or local .env"""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


def setup_logging():
    """Configure root logger for the application."""

    Path("logs").mkdir(exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


# ─── Application Settings ───

class Settings:
    """Centralized configuration loaded from environment variables."""

    # ─── Paths ───
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "models"))
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", "logs"))

    # ─── Database ───
    DB_PATH: str = os.getenv("DB_PATH", str(DATA_DIR / "crypto.db"))

    # ─── CoinGecko API ───
    COINGECKO_API_URL: str = os.getenv("COINGECKO_API_URL", "https://api.coingecko.com/api/v3")
    COINGECKO_API_KEY: Optional[str] = os.getenv("COINGECKO_API_KEY")
    FETCH_INTERVAL_MINUTES: int = int(os.getenv("FETCH_INTERVAL_MINUTES", "60"))

    # ─── Model ───
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v1.0.0")
    SEQUENCE_LENGTH: int = int(os.getenv("SEQUENCE_LENGTH", "24"))
    FEATURE_COLS: list[str] = [
        "rsi", "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_lower", "bb_percent",
        "price_change_1h", "price_change_24h", "volatility",
    ]

    # ─── Ollama / LLM ───
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))
    OLLAMA_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    OLLAMA_RETRY_BACKOFF: float = float(os.getenv("OLLAMA_RETRY_BACKOFF", "2.0"))
    LLM_CACHE_TTL_HOURS: int = int(os.getenv("LLM_CACHE_TTL_HOURS", "1"))

    # ─── Drift Detection ───
    DRIFT_WINDOW_SIZE: int = int(os.getenv("DRIFT_WINDOW_SIZE", "100"))
    DRIFT_PSI_THRESHOLD: float = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))
    DRIFT_KS_THRESHOLD: float = float(os.getenv("DRIFT_KS_THRESHOLD", "0.1"))

    # ─── Backtest ───
    BACKTEST_TRAIN_SPLIT: float = float(os.getenv("BACKTEST_TRAIN_SPLIT", "0.8"))

    # ─── Deployment ───
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    ENV: str = os.getenv("ENV", "development")

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories on startup."""
        for d in (cls.DATA_DIR, cls.MODELS_DIR, cls.LOGS_DIR):
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()