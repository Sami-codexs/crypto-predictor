import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from src.fetch_data import CoinGeckoFetcher
from src.indicators import TechnicalIndicators
import time
from src.validation import DataValidator

logger = logging.getLogger(__name__)


class CryptoScheduler:
    """
    Automated scheduler for crypto data pipeline.
    Runs fetch + indicator calculation every hour.
    """
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.fetcher = CoinGeckoFetcher()
        self.indicators = TechnicalIndicators()
        
        # Add listeners for monitoring
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
    
    def fetch_and_process_job(self):
            """The scheduled job with validation."""
            logger.info("=" * 50)
            logger.info("Starting scheduled job: fetch_and_process")
            
            try:
                # Fetch with validation built-in
                result = self.fetcher.fetch_and_store(["bitcoin", "ethereum"])
                logger.info(f"Fetch result: {result}")
                
                if result["records_stored"] == 0:
                    logger.error("No valid records stored, skipping indicator calculation")
                    return
                
                # Validate recent data before calculating indicators
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
                
                logger.info("Scheduled job completed successfully")
                
            except Exception as e:
                logger.error(f"Scheduled job failed: {e}")
    
    def start(self):
        """Start the scheduler."""
        # Schedule job every hour
        self.scheduler.add_job(
            self.fetch_and_process_job,
            'interval',
            hours=1,
            id='crypto_pipeline',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Scheduler started. Running every 1 hour.")
        
        # Keep main thread alive
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
            logger.error(f"Data validation failed for {coin_id}: {len(errors)} errors, fresh={is_fresh}")
            return False
        
        logger.info(f"Data validation passed for {coin_id}: {len(cleaned_df)} valid rows")
        return True


def run_once():
    """Run pipeline once immediately (for testing)."""
    sched = CryptoScheduler()
    sched.fetch_and_process_job()


if __name__ == "__main__":
    from config import setup_logging
    setup_logging()
    
    logger.info("Starting Crypto Scheduler")
    scheduler = CryptoScheduler()
    scheduler.start()
