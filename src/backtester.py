import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulates trading strategy on historical data.
    Calculates returns, risk metrics, performance analytics.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, df: pd.DataFrame, model: CryptoLSTM, 
                     threshold: float = 0.6, fee_pct: float = 0.001) -> Dict:
        """
        Run walk-forward backtest.
        
        Strategy:
        - Predict next hour
        - If confidence > threshold and prediction = 'up', buy
        - Hold for 1 hour, then sell
        """
        logger.info(f"Starting backtest: ${self.initial_capital:,.2f} capital")
        
        # Create sequences
        preprocessor = MLPreprocessor(sequence_length=24)
        
        # Need to manually create sequences for backtest
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'target', 'coin_id']]
        
        # Walk forward: for each possible prediction point
        for i in range(24, len(df) - 1):  # Start after first 24h
            # Get sequence ending at i
            sequence = df[feature_cols].iloc[i-24:i].values
            
            # Current price (entry)
            entry_price = df['price_usd'].iloc[i]
            timestamp = df['timestamp'].iloc[i]
            
            # Next price (exit - 1 hour later)
            exit_price = df['price_usd'].iloc[i+1]
            
            # Make prediction
            X = np.array([sequence])
            try:
                prob = model.predict(X)[0]
            except Exception as e:
                logger.warning(f"Prediction failed at {timestamp}: {e}")
                continue
            
            prediction = 'up' if prob > 0.5 else 'down'
            confidence = prob if prediction == 'up' else (1 - prob)
            
            # Trading logic: Only buy if confident up
            if prediction == 'up' and confidence >= threshold:
                # Simulate trade
                position_size = self.capital  # All-in for simplicity
                
                # Entry with fee
                shares = position_size * (1 - fee_pct) / entry_price
                entry_value = shares * entry_price
                
                # Exit after 1 hour with fee
                exit_value = shares * exit_price * (1 - fee_pct)
                profit = exit_value - position_size
                profit_pct = profit / position_size
                
                # Update capital
                self.capital = exit_value
                
                trade = {
                    'timestamp': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'confidence': confidence,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'capital': self.capital
                }
                self.trades.append(trade)
                
                logger.debug(f"Trade at {timestamp}: {profit_pct:+.2%}")
            
            # Record equity
            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital
            })
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        profits = [t['profit_pct'] for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        win_rate = len(wins) / len(profits) if profits else 0
        
        # Risk metrics
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float('inf')
        
        # Drawdown
        equity_values = [e['capital'] for e in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365 * 24) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        metrics = {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': self.capital,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe
        }
        
        logger.info(f"Backtest complete: {metrics['total_trades']} trades, "
                   f"{win_rate:.1%} win rate, {total_return:+.2%} return")
        
        return metrics
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log."""
        return pd.DataFrame(self.trades)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve over time."""
        return pd.DataFrame(self.equity_curve)