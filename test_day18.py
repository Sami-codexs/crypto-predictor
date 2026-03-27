#!/usr/bin/env python3
"""
Day 18 Test: Data Drift Detection
"""

import sys
sys.path.insert(0, 'src')

from config import setup_logging
from indicators import TechnicalIndicators
from drift_detector import DriftDetector, PredictionDriftMonitor
import numpy as np

setup_logging()

print("=" * 70)
print("Day 18 Test: Data Drift Detection")
print("=" * 70)

# Load data
indicators = TechnicalIndicators()

try:
    df_all = indicators.engineer_features("bitcoin", hours=277)  # Use what you have
    # Split: ~50/50 or 60/40
    split_point = len(df_all) // 2  # Split in middle
    df_ref = df_all.iloc[:split_point].copy()      # ~138 records
    df_current = df_all.iloc[split_point:].copy()  # ~139 records
    
    print(f"   Reference: {len(df_ref)} samples")
    print(f"   Current: {len(df_current)} samples")
    
except ValueError as e:
    print(f"❌ Error: {e}")
    print("   Need 300+ hours of data. Run data collection first.")
    exit(1)

# Test Drift Detection
print("\n" + "-" * 70)
print("Test: Detect Drift")
print("-" * 70)

detector = DriftDetector(reference_data=df_ref)
results = detector.detect_drift(df_current)

print(f"\n📊 Status: {'⚠️ DRIFT' if results['drift_detected'] else '✅ OK'}")
print(f"📊 Drifted: {len(results['drifted_features'])}/{len(results['features'])} features")
print(f"\n📝 {results['recommendation']}")

# Show top 3 drifted features
feature_scores = []
for feat, metrics in results['features'].items():
    score = metrics['psi'] + (1 - metrics['ks_pvalue'])
    feature_scores.append((feat, score, metrics))

feature_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 3 Most Drifted Features:")
for i, (feat, score, metrics) in enumerate(feature_scores[:3], 1):
    status = "⚠️" if metrics['drift_detected'] else "✅"
    print(f"\n{i}. {feat} {status}")
    print(f"   PSI: {metrics['psi']:.4f}")
    print(f"   K-S p: {metrics['ks_pvalue']:.4f}")

# Test Prediction Drift
print("\n" + "-" * 70)
print("Test: Prediction Drift Monitor")
print("-" * 70)

pred_monitor = PredictionDriftMonitor()
np.random.seed(42)
baseline_preds = np.random.beta(2, 2, 1000)
pred_monitor.set_baseline(baseline_preds)

current_preds = np.random.beta(3, 2, 200)
pred_drift = pred_monitor.check_prediction_drift(current_preds)

print(f"\n📊 Prediction Drift: {'⚠️' if pred_drift['drift_detected'] else '✅'}")
print(f"   K-S p-value: {pred_drift['ks_pvalue']:.4f}")
print(f"   Mean shift: {pred_drift['mean_shift']:+.4f}")

print("\n" + "=" * 70)
print("✅ Day 18 Complete!")
print("=" * 70)