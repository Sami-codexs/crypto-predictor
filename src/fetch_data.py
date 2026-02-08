import requests
import time
from datetime import datetime
from database import CryptoDatabase
import logging

logger = logging.getLogger(__name__)

class CoinGeckoFetcher:
    """
    Fetches crypto data from CoinGecko API.
    Free tier: 50 calls/minute, no API key needed (perfect for learning)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self.db = CryptoDatabase()
        self.session = requests.Session()  # Reuse connections (performance)
    
    def fetch_current_price(self, coin_ids: list) -> dict:
        """
        Fetch current price for multiple coins.
        Returns: {coin_id: {usd: price, volume: vol, market_cap: cap}}
        """
        # CoinGecko expects comma-separated string
        coins_str = ",".join(coin_ids)
        
        endpoint = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coins_str,
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_market_cap": "true",
            "include_last_updated_at": "true"
        }
        
        try:
            # Interview tip: Always set timeout on external APIs
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()  # Raises exception for 4XX/5XX
            
            data = response.json()
            self.db.log_fetch(len(coin_ids), "success")
            logger.info(f"Fetched prices for {len(coin_ids)} coins")
            return data
            
        except requests.exceptions.RequestException as e:
            self.db.log_fetch(len(coin_ids), "failed", str(e))
            logger.error(f"API request failed: {e}")
            raise  # Let caller handle retry logic
    
    def store_prices(self, price_data: dict):
        """Parse API response and store in database"""
        stored_count = 0
        
        for coin_id, data in price_data.items():
            success = self.db.insert_price(
                coin_id=coin_id,
                price=data.get("usd"),
                volume=data.get("usd_24h_vol"),
                market_cap=data.get("usd_market_cap")
            )
            if success:
                stored_count += 1
        
        logger.info(f"Stored {stored_count} new price records")
        return stored_count
    
    def fetch_and_store(self, coin_ids: list = None):
        """
        Main method: Fetch + Store in one operation.
        This is what runs every hour via scheduler.
        """
        if coin_ids is None:
            coin_ids = ["bitcoin", "ethereum"]  # Default portfolio
        
        try:
            price_data = self.fetch_current_price(coin_ids)
            count = self.store_prices(price_data)
            return {"status": "success", "records_stored": count}
            
        except Exception as e:
            logger.error(f"Fetch and store failed: {e}")
            return {"status": "error", "message": str(e)}


# Standalone execution for testing
if __name__ == "__main__":
    print("🚀 Starting Day 1: Data Fetcher Test")
    print("=" * 50)
    
    fetcher = CoinGeckoFetcher()
    
    # Test with Bitcoin only first
    result = fetcher.fetch_and_store(["bitcoin"])
    
    print(f"\nResult: {result}")
    
    # Verify data was stored
    recent = fetcher.db.get_recent_prices("bitcoin", hours=1)
    print(f"\nRecent data in DB:\n{recent.head()}")
    
    print("\n✅ Day 1 Complete! Data pipeline working.")