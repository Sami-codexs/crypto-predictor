import pandas as pd
import numpy as np
import logging
from database import CryptoDatabase

# Get logger for this file
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """
    This class calculates trading signals from price data.
    Think of it as a "smart calculator" for Bitcoin prices.
    """
    
    def __init__(self, db: CryptoDatabase = None):
        # Connect to database (or create new connection)
        self.db = db or CryptoDatabase()
    
    # ============================================================
    # INDICATOR 1: RSI (Relative Strength Index)
    # ============================================================
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        RSI tells you if Bitcoin is "overbought" (too expensive) or "oversold" (too cheap)
        
        Scale: 0 to 100
        - RSI > 70 = Overbought (might go down soon) 🔴
        - RSI < 30 = Oversold (might go up soon) 🟢
        - 30-70 = Normal range 🟡
        
        Why 14 periods? It's the standard that traders have used for 50 years.
        """
        
        # Step 1: Calculate price changes (today - yesterday)
        # Example: [100, 102, 101] → [NaN, +2, -1]
        delta = prices.diff()
        
        # Step 2: Separate UP moves and DOWN moves
        gains = delta.where(delta > 0, 0)      # Only positive changes
        losses = -delta.where(delta < 0, 0)    # Only negative changes (made positive)
        
        # Step 3: Calculate average gains and losses over last 14 periods
        # Using "exponential moving average" - gives more weight to recent prices
        avg_gains = gains.ewm(alpha=1/period, min_periods=period).mean()
        avg_losses = losses.ewm(alpha=1/period, min_periods=period).mean()
        
        # Step 4: Calculate Relative Strength (RS)
        # RS = how strong are gains compared to losses?
        rs = avg_gains / avg_losses
        
        # Step 5: Convert to 0-100 scale
        rsi = 100 - (100 / (1 + rs))
        
        logger.debug(f"RSI calculated: {rsi.iloc[-1]:.2f} (latest value)")
        return rsi
    
    # ============================================================
    # INDICATOR 2: MACD (Moving Average Convergence Divergence)
    # ============================================================
    
    def calculate_macd(self, prices: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """
        MACD tells you if the TREND is changing.
        
        Think of it like two cars racing:
        - Fast car (12-period average) 
        - Slow car (26-period average)
        
        When fast car crosses ABOVE slow car = Bullish (price might go up) 🚀
        When fast car crosses BELOW slow car = Bearish (price might go down) 📉
        """
        
        # Calculate "exponential moving averages" (EMA)
        # EMA = weighted average where recent prices count more
        ema_fast = prices.ewm(span=fast, adjust=False).mean()   # 12-period
        ema_slow = prices.ewm(span=slow, adjust=False).mean()   # 26-period
        
        # MACD Line = difference between fast and slow
        macd_line = ema_fast - ema_slow
        
        # Signal Line = 9-period average of MACD Line (the "slow car" for MACD)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram = how far apart they are (bigger = stronger signal)
        histogram = macd_line - signal_line
        
        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
    
    # ============================================================
    # INDICATOR 3: Bollinger Bands
    # ============================================================
    
    def calculate_bollinger_bands(self, prices: pd.Series, 
                                  period: int = 20, std_dev: int = 2) -> dict:
        """
        Bollinger Bands create a "channel" around the price.
        
        Imagine a river:
        - Middle band = the river's center (average price)
        - Upper band = the river bank (price rarely goes above)
        - Lower band = the other river bank (price rarely goes below)
        
        When price touches the upper band = might be too high
        When price touches the lower band = might be too low
        """
        
        # Middle band = simple moving average (20 periods)
        sma = prices.rolling(window=period).mean()
        
        # Calculate standard deviation (how much price varies)
        rolling_std = prices.rolling(window=period).std()
        
        # Upper band = average + 2 standard deviations
        upper_band = sma + (rolling_std * std_dev)
        
        # Lower band = average - 2 standard deviations
        lower_band = sma - (rolling_std * std_dev)
        
        # %B = where is price within the bands? (0 = bottom, 1 = top, 0.5 = middle)
        percent_b = (prices - lower_band) / (upper_band - lower_band)
        
        return {
            'middle_band': sma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'percent_b': percent_b
        }
    
    # ============================================================
    # MAIN METHOD: Put It All Together
    # ============================================================
    
    def engineer_features(self, coin_id: str, hours: int = 48) -> pd.DataFrame:
        """
        This is the "master function" that:
        1. Gets raw prices from database
        2. Calculates all indicators
        3. Returns a table ready for machine learning
        
        Think of it as a factory: raw materials → finished product
        """
        
        logger.info(f"Creating features for {coin_id} (last {hours} hours)")
        
        # Step 1: Get raw data from database (from Day 1)
        df = self.db.get_recent_prices(coin_id, hours=hours)
        
        # Safety check: need at least 26 data points for MACD
        if len(df) < 26:
            raise ValueError(f"Need 26+ prices, only have {len(df)}. Run fetcher more times!")
        
        # Sort by time (oldest first)
        df = df.sort_values('timestamp')
        
        # Step 2: Calculate RSI (add as new column)
        df['rsi'] = self.calculate_rsi(df['price_usd'])
        
        # Step 3: Calculate MACD (adds 3 new columns)
        macd_data = self.calculate_macd(df['price_usd'])
        df['macd_line'] = macd_data['macd_line']
        df['macd_signal'] = macd_data['signal_line']
        df['macd_histogram'] = macd_data['histogram']
        
        # Step 4: Calculate Bollinger Bands (adds 4 new columns)
        bb_data = self.calculate_bollinger_bands(df['price_usd'])
        df['bb_upper'] = bb_data['upper_band']
        df['bb_lower'] = bb_data['lower_band']
        df['bb_percent'] = bb_data['percent_b']
        
        # Step 5: Add extra "smart" features
        # Price change in last 1 hour (percentage)
        df['price_change_1h'] = df['price_usd'].pct_change(periods=1)
        
        # Price change in last 24 hours
        df['price_change_24h'] = df['price_usd'].pct_change(periods=24)
        
        # Volatility = how much price jumps around (standard deviation)
        df['volatility'] = df['price_usd'].rolling(window=24).std()
        
        # Step 6: Create TARGET variable (what we want to predict)
        # Target = 1 if price goes UP in next hour, 0 if DOWN
        # .shift(-1) looks at NEXT row (future price)
        df['target'] = (df['price_usd'].shift(-1) > df['price_usd']).astype(int)
        
        logger.info(f"Features created: {len(df.columns)} columns, {len(df)} rows")
        return df
    
    # ============================================================
    # HELPER: Get Simple Trading Signals
    # ============================================================
    
    def get_latest_signals(self, coin_id: str) -> dict:
        """
        Convert numbers to human-readable trading advice.
        
        Instead of "RSI: 75", say "RSI: 75 (overbought - might drop)"
        """
        
        # Get all features
        df = self.engineer_features(coin_id, hours=48)
        
        # Get the most recent row (latest data)
        latest = df.iloc[-1]
        
        # Interpret RSI
        if latest['rsi'] > 70:
            rsi_signal = "overbought"
        elif latest['rsi'] < 30:
            rsi_signal = "oversold"
        else:
            rsi_signal = "neutral"
        
        # Interpret MACD (look for crossovers)
        prev_histogram = df.iloc[-2]['macd_histogram']
        curr_histogram = latest['macd_histogram']
        
        if curr_histogram > 0 and prev_histogram <= 0:
            macd_signal = "bullish_crossover"  # Just turned positive! 🚀
        elif curr_histogram < 0 and prev_histogram >= 0:
            macd_signal = "bearish_crossover"  # Just turned negative! 📉
        elif curr_histogram > 0:
            macd_signal = "bullish"
        else:
            macd_signal = "bearish"
        
        return {
            "coin": coin_id,
            "timestamp": latest['timestamp'],
            "price": round(latest['price_usd'], 2),
            "rsi": round(latest['rsi'], 2),
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "macd_histogram": round(latest['macd_histogram'], 4),
            "bb_position": round(latest['bb_percent'], 4),
            "volatility_24h": round(latest['volatility'], 2) if not pd.isna(latest['volatility']) else None
        }