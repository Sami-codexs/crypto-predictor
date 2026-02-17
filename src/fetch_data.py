import requests
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.database import CryptoDatabase
from src.validation import DataValidator

logger = logging.getLogger(__name__)


class CoinGeckoFetcher:
    """
    Fetches crypto data from CoinGecko API with retries and validation.
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self):
        self.db = CryptoDatabase()
        self.validator = DataValidator()
        self.session = requests.Session()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException,)),
        reraise=True
    )
    def _make_request(self, endpoint: str, params: dict) -> dict:
        """
        Make API request with exponential backoff retry.
        - Try 3 times max
        - Wait 4s, then 8s, then 16s between retries
        """
        logger.debug(f"API request to {endpoint}")
        response = self.session.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    
    def fetch_current_price(self, coin_ids: list) -> dict:
        """
        Fetch current price with validation and error handling.
        """
        coins_str = ",".join(coin_ids)
        
        endpoint = f"{self.BASE_URL}/simple/price"
        params = {
            "ids": coins_str,
            "vs_currencies": "usd",
            "include_24hr_vol": "true",
            "include_market_cap": "true"
        }
        
        try:
            data = self._make_request(endpoint, params)
            logger.info(f"Fetched prices for {len(coin_ids)} coins")
            return data
            
        except requests.exceptions.RequestException as e:
            self.db.log_fetch(len(coin_ids), "failed", str(e))
            logger.error(f"API request failed after retries: {e}")
            raise
    
    def store_prices(self, price_data: dict) -> dict:
        """
        Store prices with validation.
        Returns summary of stored/invalid records.
        """
        results = {"stored": 0, "invalid": 0, "errors": []}
        
        for coin_id, data in price_data.items():
            price = data.get("usd")
            
            # Validate
            is_valid, msg = self.validator.validate_price(price)
            
            if not is_valid:
                logger.warning(f"Invalid price for {coin_id}: {msg}")
                results["invalid"] += 1
                results["errors"].append(f"{coin_id}: {msg}")
                continue
            
            # Store if valid
            success = self.db.insert_price(
                coin_id=coin_id,
                price=price,
                volume=data.get("usd_24h_vol"),
                market_cap=data.get("usd_market_cap")
            )
            
            if success:
                results["stored"] += 1
        
        return results
    
    def fetch_and_store(self, coin_ids: list = None) -> dict:
        """
        Main method with full error handling.
        """
        if coin_ids is None:
            coin_ids = ["bitcoin", "ethereum"]
        
        try:
            price_data = self.fetch_current_price(coin_ids)
            results = self.store_prices(price_data)
            
            self.db.log_fetch(
                coins_count=results["stored"],
                status="success" if results["stored"] > 0 else "partial",
                error="; ".join(results["errors"]) if results["errors"] else None
            )
            
            return {
                "status": "success" if results["stored"] > 0 else "failed",
                "records_stored": results["stored"],
                "records_invalid": results["invalid"],
                "errors": results["errors"]
            }
            
        except Exception as e:
            self.db.log_fetch(len(coin_ids), "failed", str(e))
            logger.error(f"Fetch and store failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "records_stored": 0
            }