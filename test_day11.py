from src.config import setup_logging
from src.indicators import TechnicalIndicators
from src.model import CryptoLSTM
from src.backtester import Backtester
from src.model_manager import ModelManager
import os

setup_logging()

print("Day 11 Test: Backtesting Framework")
print("=" * 50)

# Load best model
manager = ModelManager()
model_path = "models/crypto_lstm_20260224_225513.keras" 
if not model_path:
    print("No model found")
    exit()

print(f"\n1. Loading model: {os.path.basename(model_path)}")
model = CryptoLSTM(sequence_length=24, n_features=12)
model.load(model_path)

# Get historical data
print("\n2. Loading historical data...")
indicators = TechnicalIndicators()
df = indicators.engineer_features("bitcoin", hours=168)  # 3 days for test

print(f"   Data: {len(df)} rows")

# Run backtest
print("\n3. Running backtest (51% confidence threshold)...")
backtester = Backtester(initial_capital=10000.0)
metrics = backtester.run_backtest(df, model, threshold=0.53, fee_pct=0.001)

if 'error' in metrics:
    print(f"   Error: {metrics['error']}")
else:
    print(f"\n4. Results:")
    print(f"   Trades: {metrics['total_trades']}")
    print(f"   Win rate: {metrics['win_rate']:.1%}")
    print(f"   Total return: {metrics['total_return_pct']:+.2f}%")
    print(f"   Final capital: ${metrics['final_capital']:,.2f}")
    print(f"   Max drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Profit factor: {metrics['profit_factor']:.2f}")

    # Show recent trades
    if metrics['total_trades'] > 0:
        print(f"\n5. Last 3 trades:")
        trades = backtester.get_trade_log().tail(3)
        for _, trade in trades.iterrows():
            print(f"   {trade['timestamp']}: {trade['profit_pct']:+.2%} (conf: {trade['confidence']:.2f})")

print("\n" + "=" * 50)
print("Day 11 complete!")
