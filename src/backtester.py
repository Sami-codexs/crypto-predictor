import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulates trading strategy on historical data.
    Calculates returns, risk metrics, and classification performance.
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []
        # ─── Classification tracking ───
        self.y_true: List[int] = []
        self.y_pred: List[int] = []

    def run_backtest(
        self,
        df: pd.DataFrame,
        model: CryptoLSTM,
        threshold: float = 0.6,
        fee_pct: float = 0.001,
    ) -> Dict:
        """
        Run walk-forward backtest with full classification metrics.

        Strategy:
        - Predict next hour
        - If confidence > threshold and prediction = 'up', buy
        - Hold for 1 hour, then sell
        """
        logger.info(f"Starting backtest: ${self.initial_capital:,.2f} capital")

        feature_cols = [
            col for col in df.columns
            if col not in ['timestamp', 'target', 'coin_id']
        ]

        sequence_length = getattr(model, 'sequence_length', 24)

        for i in range(sequence_length, len(df) - 1):
            sequence = df[feature_cols].iloc[i - sequence_length:i].values
            entry_price = df['price_usd'].iloc[i]
            timestamp = df['timestamp'].iloc[i]
            exit_price = df['price_usd'].iloc[i + 1]

            # Ground-truth: did price go up over the next hour?
            actual_up = int(exit_price > entry_price)

            X = np.array([sequence])
            try:
                prob = model.predict(X)[0]
            except Exception as e:
                logger.warning(f"Prediction failed at {timestamp}: {e}")
                continue

            prediction = 1 if prob > 0.5 else 0
            confidence = prob if prediction == 1 else (1 - prob)

            # ─── Classification bookkeeping ───
            self.y_true.append(actual_up)
            self.y_pred.append(prediction)

            # Trading logic: Only buy if confident up
            if prediction == 1 and confidence >= threshold:
                position_size = self.capital
                shares = position_size * (1 - fee_pct) / entry_price
                exit_value = shares * exit_price * (1 - fee_pct)
                profit = exit_value - position_size
                profit_pct = profit / position_size

                self.capital = exit_value

                trade = {
                    'timestamp': timestamp,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'confidence': confidence,
                    'predicted_up': prediction,
                    'actual_up': actual_up,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'capital': self.capital,
                }
                self.trades.append(trade)
                logger.debug(f"Trade at {timestamp}: {profit_pct:+.2%}")

            self.equity_curve.append({
                'timestamp': timestamp,
                'capital': self.capital,
            })

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """Calculate trading + classification performance metrics."""
        metrics: Dict = {}

        # ─── Classification Metrics ───
        if self.y_true and self.y_pred:
            metrics['classification'] = {
                'accuracy': float(accuracy_score(self.y_true, self.y_pred)),
                'precision': float(precision_score(self.y_true, self.y_pred, zero_division=0)),
                'recall': float(recall_score(self.y_true, self.y_pred, zero_division=0)),
                'f1_score': float(f1_score(self.y_true, self.y_pred, zero_division=0)),
                'confusion_matrix': confusion_matrix(self.y_true, self.y_pred).tolist(),
                'total_predictions': len(self.y_true),
                'actual_up_count': int(sum(self.y_true)),
                'actual_down_count': int(len(self.y_true) - sum(self.y_true)),
            }
        else:
            metrics['classification'] = {'error': 'No predictions made'}

        # ─── Trading Metrics ───
        if not self.trades:
            metrics['trading'] = {'error': 'No trades executed'}
            return metrics

        profits = [t['profit_pct'] for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        win_rate = len(wins) / len(profits) if profits else 0

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = (
            abs(sum(wins) / sum(losses))
            if losses and sum(losses) != 0
            else float('inf')
        )

        equity_values = [e['capital'] for e in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0.0

        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(365 * 24)
            if len(returns) > 1 and np.std(returns) > 0
            else 0.0
        )

        metrics['trading'] = {
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
            'sharpe_ratio': sharpe,
        }

        logger.info(
            f"Backtest complete: {metrics['trading']['total_trades']} trades, "
            f"{win_rate:.1%} win rate, {total_return:+.2%} return | "
            f"Accuracy: {metrics['classification'].get('accuracy', 0):.1%}"
        )

        return metrics

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log."""
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve over time."""
        return pd.DataFrame(self.equity_curve)

    def get_classification_report(self) -> str:
        """Human-readable classification summary."""
        if not self.y_true:
            return "No predictions were made during the backtest."

        from sklearn.metrics import classification_report as sk_report
        return sk_report(
            self.y_true,
            self.y_pred,
            target_names=['Down', 'Up'],
            digits=3,
        )