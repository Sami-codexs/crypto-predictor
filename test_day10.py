from src.config import setup_logging
from src.model_manager import ModelManager
from src.model import CryptoLSTM
from src.preprocessing import MLPreprocessor
from src.indicators import TechnicalIndicators
import os

setup_logging()

print("Day 10 Test: Model Management & Versioning")
print("=" * 50)

# Initialize manager
manager = ModelManager()

# List all models
print("\n1. Available models:")
models = manager.list_models()
for i, model in enumerate(models[:5], 1):
    acc = f"{model['accuracy']:.3f}" if model['accuracy'] else "N/A"
    print(f"   {i}. {model['name']} (acc: {acc})")

# Get best model
print("\n2. Selecting best model...")
best_path = manager.get_best_model('accuracy')
if best_path:
    print(f"   Best: {os.path.basename(best_path)}")
else:
    print("   No models found")

# Get latest model
print("\n3. Latest model:")
latest_path = manager.get_latest_model()
if latest_path:
    print(f"   Latest: {os.path.basename(latest_path)}")

# Test performance logging
print("\n4. Logging mock performance...")
test_metrics = {
    'accuracy': 0.588,
    'precision': 0.80,
    'recall': 0.40,
    'f1_score': 0.533,
    'test_samples': 17
}
manager.log_performance("crypto_lstm_20260224_225513.keras", test_metrics, "Day 8 training")

# Retrieve performance history
print("\n5. Performance history:")
history = manager.get_performance_history()
for record in history[:3]:
    print(f"   {record['model_name']}: acc={record['accuracy']:.3f} at {record['timestamp']}")

# Comparison report
if len(models) >= 1:
    print("\n6. Model comparison:")
    model_names = [m['name'] for m in models[:2]]
    report = manager.compare_models(model_names)
    print(report)

print("\n" + "=" * 50)
print("Day 10 complete!")