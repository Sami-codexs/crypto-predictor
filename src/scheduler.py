import logging
import time
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from src.fetch_data import CoinGeckoFetcher
from src.indicators import TechnicalIndicators
from src.config import settings
from src.database import CryptoDatabase
from src.drift_detector import DriftDetector
from src.validation import DataValidator

logger = logging.getLogger(__name__)


class CryptoScheduler:
    """
    Automated scheduler for crypto data pipeline.
    Runs fetch + indicator calculation + drift detection every hour.
    """

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.fetcher = CoinGeckoFetcher()
        self.indicators = TechnicalIndicators()
        self.db = CryptoDatabase(settings.DB_PATH)
        self.drift_detector: DriftDetector = None
        self.reference_data = None

        self.scheduler.add_listener(
            self._job_listener,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

    def _job_listener(self, event):
        """Log job execution results."""
        if event.exception:
            logger.error(f"Job {event.job_id} crashed: {event.exception}")
        else:
            logger.info(f"Job {event.job_id} completed")

    def _load_reference_for_drift(self):
        """Load or refresh the reference dataset for drift detection."""
        try:
            # Pull the oldest N rows from indicators as reference
            df = self.db.get_indicator_window(
                coin_id="bitcoin",
                lookback=settings.DRIFT_WINDOW_SIZE * 2,
            )
            if len(df) >= settings.DRIFT_WINDOW_SIZE:
                self.reference_data = df.tail(settings.DRIFT_WINDOW_SIZE)
                if self.drift_detector is None:
                    self.drift_detector = DriftDetector(
                        reference_data=self.reference_data,
                        db=self.db,
                    )
                else:
                    self.drift_detector.set_reference(self.reference_data)
                logger.info(
                    f"Drift reference loaded: {len(self.reference_data)} samples"
                )
        except Exception as e:
            logger.warning(f"Could not load drift reference: {e}")

    def fetch_and_process_job(self):
        """The scheduled job with validation and drift detection."""
        logger.info("=" * 50)
        logger.info("Starting scheduled job: fetch_and_process")

        try:
            # ─── 1. Fetch ───
            result = self.fetcher.fetch_and_store(["bitcoin", "ethereum"])
            logger.info(f"Fetch result: {result}")

            if result.get("records_stored", 0) == 0:
                logger.error("No valid records stored, skipping indicator calculation")
                return

            # ─── 2. Indicators + Validation ───
            for coin in ["bitcoin", "ethereum"]:
                if not self.validate_recent_data(coin):
                    logger.warning(f"Skipping {coin} due to validation failures")
                    continue

                try:
                    df = self.indicators.engineer_features(coin, hours=48)
                    self.indicators.db.save_indicators(coin, df)
                    logger.info(f"Indicators updated for {coin}: {len(df)} rows")
                except ValueError as e:
                    logger.warning(f"Not enough data for {coin}: {e}")
                    continue

                # ─── 3. Drift Detection ───
                try:
                    self._run_drift_check(coin, df)
                except Exception as e:
                    logger.error(f"Drift check failed for {coin}: {e}")

            logger.info("Scheduled job completed successfully")

        except Exception as e:
            logger.error(f"Scheduled job failed: {e}")

    def _run_drift_check(self, coin_id: str, current_df):
        """Run drift detection if reference data is available."""
        if self.drift_detector is None:
            self._load_reference_for_drift()

        if self.drift_detector is None or self.drift_detector.reference_data is None:
            logger.info("Drift detector not ready, skipping check")
            return

        # Use only feature columns for drift
        feature_cols = [
            c for c in current_df.columns
            if c not in ['timestamp', 'target', 'coin_id']
        ]
        current_features = current_df[feature_cols].dropna()

        if len(current_features) < 10:
            logger.warning("Insufficient rows for drift detection")
            return

        results = self.drift_detector.detect_drift(current_features)

        if results['drift_detected']:
            logger.warning(
                f"DRIFT ALERT for {coin_id}: "
                f"{len(results['drifted_features'])} features drifted | "
                f"{results['recommendation']}"
            )
        else:
            logger.info(f"Drift check passed for {coin_id}")

    def start(self):
        """Start the scheduler."""
        self.scheduler.add_job(
            self.fetch_and_process_job,
            'interval',
            minutes=settings.FETCH_INTERVAL_MINUTES,
            id='crypto_pipeline',
            replace_existing=True,
        )

        # Also schedule a reference refresh every 6 hours
        self.scheduler.add_job(
            self._load_reference_for_drift,
            'interval',
            hours=6,
            id='drift_reference_refresh',
            replace_existing=True,
        )

        self.scheduler.start()
        logger.info(
            f"Scheduler started. Running every {settings.FETCH_INTERVAL_MINUTES} min."
        )

        try:
            while True:
                time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        """Stop the scheduler gracefully."""
        self.scheduler.shutdown()
        logger.info("Scheduler stopped")

    def validate_recent_data(self, coin_id: str) -> bool:
        """Check if recent data passes validation."""
        df = self.indicators.db.get_recent_prices(coin_id, hours=24)
        validator = DataValidator()

        is_valid, cleaned_df, errors = validator.validate_dataframe(df)
        is_fresh = validator.check_data_freshness(df, max_age_hours=2)

        if not is_valid or not is_fresh:
            logger.error(
                f"Data validation failed for {coin_id}: "
                f"{len(errors)} errors, fresh={is_fresh}"
            )
            return False

        logger.info(
            f"Data validation passed for {coin_id}: {len(cleaned_df)} valid rows"
        )
        return True


def run_once():
    """Run pipeline once immediately (for testing)."""
    sched = CryptoScheduler()
    sched._load_reference_for_drift()
    sched.fetch_and_process_job()


if __name__ == "__main__":
    from config import setup_logging

    setup_logging()
    logger.info("Starting Crypto Scheduler")
    scheduler = CryptoScheduler()
    scheduler.start()