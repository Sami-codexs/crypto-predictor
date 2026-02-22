from src.config import setup_logging
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM
import numpy as np

setup_logging()

print("Day 8 Test: Improved Training with Augmentation")
print("=" * 50)

# Get data
indicators = TechnicalIndicators()
df = indicators.engineer_features("bitcoin", hours=72)

print(f"\n1. Raw data: {len(df)} rows")

# Preprocess with overlapping sequences
preprocessor = MLPreprocessor(sequence_length=24)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df, use_overlap=True)

print(f"\n2. Augmented data:")
print(f"   X_train: {X_train.shape}")
print(f"   X_test:  {X_test.shape}")
print(f"   Class balance - train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")

# Build model
print("\n3. Building model...")
model = CryptoLSTM(sequence_length=24, n_features=X_train.shape[2])
model.build_model(lstm_units=50, dropout_rate=0.3)  # Higher dropout

# Train with early stopping
print("\n4. Training with early stopping (max 100 epochs)...")
metrics = model.train(X_train, y_train, X_test, y_test, 
                      epochs=100, batch_size=4, use_early_stopping=True)

print(f"\n5. Training results:")
print(f"   Epochs: {metrics['epochs_trained']}")
print(f"   Train accuracy: {metrics['final_accuracy']:.3f}")
print(f"   Val accuracy: {metrics['final_val_accuracy']:.3f}")

# Evaluate
print("\n6. Final evaluation...")
eval_metrics = model.evaluate(X_test, y_test)
print(f"   Test accuracy: {eval_metrics['accuracy']:.3f}")
print(f"   Precision: {eval_metrics['precision']:.3f}")
print(f"   Recall: {eval_metrics['recall']:.3f}")
print(f"   F1: {eval_metrics['f1_score']:.3f}")

# Save if decent
if eval_metrics['accuracy'] >= 0.55:
    print("\n7. Saving model...")
    filepath = model.save()
    print(f"   Saved: {filepath}")
else:
    print("\n7. Model accuracy < 55%, not saving. Collect more data.")

print("\n" + "=" * 50)
print("Day 8 complete!")