from src.config import setup_logging
from src.indicators import TechnicalIndicators
from src.preprocessing import MLPreprocessor
from src.model import CryptoLSTM

setup_logging()

print("Day 7 Test: LSTM Model Architecture")
print("=" * 50)

# Get data
indicators = TechnicalIndicators()
df = indicators.engineer_features("bitcoin", hours=72)
print(f"\n1. Data: {len(df)} rows")

# Preprocess
preprocessor = MLPreprocessor(sequence_length=24)
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
print(f"\n2. Preprocessed: X_train{X_train.shape}, X_test{X_test.shape}")

# Build model
print("\n3. Building LSTM...")
model = CryptoLSTM(sequence_length=24, n_features=X_train.shape[2])
model.build_model(lstm_units=50, dropout_rate=0.2)
model.model.summary()

# Train
print("\n4. Training (10 epochs for test)...")
metrics = model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=2)
print(f"   Accuracy: {metrics['final_accuracy']:.3f}")

# Evaluate
print("\n5. Evaluating...")
eval_metrics = model.evaluate(X_test, y_test)
print(f"   Test accuracy: {eval_metrics['accuracy']:.3f}")

# Save
print("\n6. Saving...")
filepath = model.save()
print(f"   Saved: {filepath}")

print("\n" + "=" * 50)
print("Day 7 complete!")