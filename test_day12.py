from src.config import setup_logging
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.optimizer import ModelOptimizer
from src.model import CryptoLSTM
import tensorflow as tf

setup_logging()

print("Day 12: Model Optimization")
print("=" * 50)

# Load data
indicators = TechnicalIndicators()
df = indicators.engineer_features("bitcoin", hours=168)  # 7 days

print(f"\n1. Data: {len(df)} rows")

# Preprocess
preprocessor = MLPreprocessor(sequence_length=24)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, use_overlap=True)

print(f"\n2. Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   Class balance: {sum(y_train==0)} down, {sum(y_train==1)} up")

# Method 1: Class weights (fix imbalance)
print("\n3. Training with class weights...")
optimizer = ModelOptimizer()

# Manual class weights (upweight the minority class)
class_weight = {0: 1.0, 1: 1.2}  # Slight upweight for "up" class

model = optimizer.train_with_class_weights(
    X_train, y_train, X_test, y_test, 
    class_weight=class_weight
)

# Evaluate
print("\n4. Evaluating optimized model...")
eval_metrics = model.evaluate(X_test, y_test)
print(f"   Test accuracy: {eval_metrics['accuracy']:.3f}")
print(f"   Precision: {eval_metrics['precision']:.3f}")
print(f"   Recall: {eval_metrics['recall']:.3f}")

# Save if improved
if eval_metrics['accuracy'] >= 0.60:
    print("\n5. Saving improved model...")
    filepath = model.save()
    print(f"   Saved: {filepath}")
else:
    print("\n5. Accuracy < 60%, needs more tuning")

print("\n" + "=" * 50)
print("Day 12 complete!")