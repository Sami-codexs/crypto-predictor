#!/usr/bin/env python3
"""
Day 22 Test: Production Monitoring
"""

import sys
sys.path.insert(0, 'src')

import time
from monitoring import ModelMonitor, get_monitor, setup_monitoring

print("=" * 70)
print("Day 22 Test: Production Monitoring")
print("=" * 70)

# Test 1: Create monitor
print("\n1. Creating ModelMonitor...")
monitor = ModelMonitor()
print("✅ Monitor created")

# Test 2: Track predictions
print("\n2. Tracking predictions...")
for i in range(20):
    coin = "bitcoin" if i % 2 == 0 else "ethereum"
    pred = "up" if i % 3 == 0 else "down"
    conf = 0.55 + (i * 0.02)
    monitor.track_prediction(coin, pred, conf)
print("✅ Tracked 20 predictions")

# Test 3: Track latency
print("\n3. Tracking latency...")
monitor.track_latency("/predict", 0.15)
monitor.track_latency("/predict", 0.23)
monitor.track_data_fetch(0.05)
print("✅ Latency tracked")

# Test 4: Update model metrics
print("\n4. Updating model metrics...")
monitor.update_model_metrics("lstm_v1", accuracy=0.58, sharpe=1.1)
monitor.update_db_count("bitcoin", 500)
print("✅ Model metrics updated")

# Test 5: Generate metrics output
print("\n5. Generating Prometheus metrics...")
metrics = monitor.get_metrics()
metrics_str = metrics.decode('utf-8')
print(f"✅ Generated {len(metrics_str)} characters of metrics")

# Check for expected metric names
assert 'crypto_predictions_total' in metrics_str
assert 'crypto_prediction_confidence' in metrics_str
assert 'crypto_prediction_latency_seconds' in metrics_str
print("✅ All expected metrics present")

# Test 6: Timer context manager
print("\n6. Testing timer context manager...")
with monitor.timer("/predict"):
    time.sleep(0.1)
print("✅ Timer context manager working")

# Test 7: Show sample metrics
print("\n" + "=" * 70)
print("Sample Metrics Output:")
print("=" * 70)
lines = metrics_str.strip().split('\n')
for line in lines[:20]:
    print(line)
if len(lines) > 20:
    print(f"... ({len(lines) - 20} more lines)")

print("\n" + "=" * 70)
print("✅ Day 22 Test Complete!")
print("=" * 70)
print()
print("🎯 Integration:")
print("   Add to api.py:")
print("   from monitoring import setup_monitoring")
print("   setup_monitoring(app)")
print()
print("📊 Access metrics at: http://localhost:8000/metrics")
print()
print("💼 PORTFOLIO VALUE:")
print("   • Production monitoring with Prometheus")
print("   • Tracks predictions, latency, model performance")
print("   • Critical for MLOps observability")