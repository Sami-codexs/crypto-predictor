from src.config import setup_logging
from src.indicators import TechnicalIndicators

setup_logging()

print("Day 2 Test: Technical Indicators")
print("=" * 40)

ind = TechnicalIndicators()

try:
    print("\n1. Engineering features...")
    df = ind.engineer_features("bitcoin", hours=48)
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"   Columns: {list(df.columns)}")
    
    print("\n2. Sample data (last 3 rows):")
    cols = ['timestamp', 'price_usd', 'rsi', 'macd_histogram', 'target']
    print(df[cols].tail(3).to_string())
    
    print("\n3. Latest signals:")
    signals = ind.get_latest_signals("bitcoin")
    for k, v in signals.items():
        print(f"   {k}: {v}")
    
    print("\n4. Saving to database...")
    ind.db.save_indicators("bitcoin", df)
    print("   Done")
    
    print("\n" + "=" * 40)
    print("Day 2 complete")

except ValueError as e:
    print(f"Error: {e}")
    print("Run fetch_data.py multiple times to collect more data")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
