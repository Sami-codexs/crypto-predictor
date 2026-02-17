import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from src.fetch_data import CoinGeckoFetcher
from src.indicators import TechnicalIndicators
import time

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
        """
        The actual job that runs every hour.
        1. Fetch new prices
        2. Calculate indicators
        3. Log results
        """
        logger.info("=" * 50)
        logger.info("Starting scheduled job: fetch_and_process")
        
        try:
            # Step 1: Fetch prices
            result = self.fetcher.fetch_and_store(["bitcoin", "ethereum"])
            logger.info(f"Fetch result: {result}")
            
            # Step 2: Calculate indicators for each coin
            for coin in ["bitcoin", "ethereum"]:
                try:
                    df = self.indicators.engineer_features(coin, hours=48)
                    self.indicators.db.save_indicators(coin, df)
                    logger.info(f"Indicators updated for {coin}: {len(df)} rows")
                except ValueError as e:
                    logger.warning(f"Not enough data for {coin}: {e}")
            
            logger.info("Scheduled job completed successfully")
            
        except Exception as e:
            logger.error(f"Scheduled job failed: {e}")
            raise
    
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
