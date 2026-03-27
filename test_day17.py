#!/usr/bin/env python3
"""
Day 17 Test: Time Series Cross-Validation
"""

import sys
sys.path.insert(0, 'src')

from config import setup_logging
from indicators import TechnicalIndicators
from preprocessing import MLPreprocessor
from model import CryptoLSTM
from time_series_cv import TimeSeriesValidator, compare_validation_strategies

setup_logging()

print("=" * 70)
print("Day 17 Test: Time Series Cross-Validation")
print("=" * 70)

# Load data
indicators = TechnicalIndicators()
try:
    df = indicators.engineer_features("bitcoin", hours=200)
    print(f"✅ Loaded {len(df)} hours of data")
except ValueError as e:
    print(f"❌ Error: {e}")
    print("Run 'python collect_data.py' first")
    exit(1)

preprocessor = MLPreprocessor(sequence_length=24)

def model_factory():
    model = CryptoLSTM(sequence_length=24, n_features=12)
    model.build_model(lstm_units=40, dropout_rate=0.3)
    return model

# Test 1: Compare strategies
print("\n" + "-" * 70)
print("Test 1: Random Split vs Time Series CV")
print("-" * 70)

comparison = compare_validation_strategies(df, preprocessor, model_factory)

print(f"\n📈 Random Split: {comparison['random_split_accuracy']:.3f}")
print(f"📉 Time Series CV: {comparison['time_series_cv_accuracy']:.3f}")
print(f"⚠️  Bias: {comparison['difference']:.3f}")

# Test 2: 5-fold CV
print("\n" + "-" * 70)
print("Test 2: 5-Fold Cross-Validation")
print("-" * 70)

validator = TimeSeriesValidator(n_splits=5, test_size=24)
results = validator.cross_validate(df, model_factory, preprocessor)

print(f"\n📊 Mean Accuracy: {results['mean_accuracy']:.3f} (±{results['std_accuracy']:.3f})")

for fold in results['fold_results']:
    print(f"   Fold {fold['fold']}: Acc={fold['accuracy']:.3f}, F1={fold['f1']:.3f}")

print("\n" + "=" * 70)
print("✅ Day 17 Complete!")
print("=" * 70)
print("\n🎓 Key Learning: Random split overestimates by 10-20% due to data leakage")