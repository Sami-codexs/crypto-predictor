from src.config import setup_logging
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
import numpy as np

setup_logging()

print("Day 6 Test: ML Preprocessing")
print("=" * 50)

# Get data
indicators = TechnicalIndicators()
df = indicators.engineer_features("bitcoin", hours=72)

print(f"\n1. Raw data: {len(df)} rows, {len(df.columns)} columns")

# Preprocess
preprocessor = MLPreprocessor(sequence_length=24)

print("\n2. Creating sequences...")
try:
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    print(f"\n3. Results:")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")
    print(f"   y_train: {np.bincount(y_train)} (0=down, 1=up)")
    print(f"   y_test:  {np.bincount(y_test)} (0=down, 1=up)")
    
    print(f"\n4. Feature columns used: {preprocessor.feature_columns}")
    
    print("\n5. Sample sequence (first timestep):")
    print(f"   First 5 features of first sequence: {X_train[0, 0, :5]}")
    
    print("\n" + "=" * 50)
    print("Day 6 complete - Data ready for LSTM!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()