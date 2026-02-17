import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates crypto data quality.
    Detects anomalies, missing values, and suspicious price movements.
    """
    
    def __init__(self):
        self.price_change_threshold = 0.15  # 15% change in 1 hour = suspicious
        self.min_price = 1000  # BTC below $1000 is clearly wrong
        self.max_price = 200000  # BTC above $200k is suspicious
    
    def validate_price(self, price: float, previous_price: Optional[float] = None) -> Tuple[bool, str]:
        """
        Validate a single price point.
        Returns: (is_valid, error_message)
        """
        # Check for missing/None
        if price is None or pd.isna(price):
            return False, "Price is None or NaN"
        
        # Check for negative or zero
        if price <= 0:
            return False, f"Invalid price: {price} (must be positive)"
        
        # Check reasonable bounds for Bitcoin
        if price < self.min_price:
            return False, f"Price ${price:,.2f} below minimum ${self.min_price:,.2f}"
        
        if price > self.max_price:
            return False, f"Price ${price:,.2f} above maximum ${self.max_price:,.2f}"
        
        # Check for sudden massive change (if we have previous price)
        if previous_price is not None and previous_price > 0:
            change_pct = abs(price - previous_price) / previous_price
            
            if change_pct > self.price_change_threshold:
                return False, f"Suspicious {change_pct:.1%} change in 1 hour"
        
        return True, "Valid"
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, list]:
        """
        Validate entire price DataFrame.
        Returns: (is_valid, cleaned_df, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required = ['timestamp', 'price_usd']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, df, [f"Missing columns: {missing}"]
        
        # Check for empty DataFrame
        if len(df) == 0:
            return False, df, ["Empty DataFrame"]
        
        # Sort by timestamp for sequential validation
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        valid_rows = []
        prev_price = None
        
        for idx, row in df.iterrows():
            is_valid, msg = self.validate_price(row['price_usd'], prev_price)
            
            if is_valid:
                valid_rows.append(idx)
                prev_price = row['price_usd']
            else:
                errors.append(f"Row {idx} ({row['timestamp']}): {msg}")
                logger.warning(f"Invalid data: {msg} at {row['timestamp']}")
        
        # Keep only valid rows
        cleaned_df = df.loc[valid_rows].copy()
        
        if len(errors) > 0:
            logger.error(f"Data validation found {len(errors)} errors, kept {len(cleaned_df)}/{len(df)} rows")
        
        return len(errors) == 0, cleaned_df, errors
    
    def check_data_freshness(self, df: pd.DataFrame, max_age_hours: int = 2) -> bool:
        """
        Check if data is recent enough.
        Returns True if latest timestamp is within max_age_hours.
        """
        if len(df) == 0:
            return False
        
        latest = pd.to_datetime(df['timestamp'].max())
        now = pd.Timestamp.now()
        age_hours = (now - latest).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            logger.error(f"Data stale: last update {age_hours:.1f} hours ago")
            return False
        
        return True