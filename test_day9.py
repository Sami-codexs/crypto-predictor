from src.config import setup_logging
from src.predictor import CryptoPredictor
import os

setup_logging()

print("Day 9 Test: Prediction Function")
print("=" * 50)

# Find latest model
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith('.keras')]
if not model_files:
    print("No trained model found. Run test_day8.py first.")
    exit()

latest_model = sorted(model_files)[-1]
model_path = os.path.join(model_dir, latest_model)
print(f"\n1. Using model: {latest_model}")

# Create predictor
print("\n2. Loading predictor...")
predictor = CryptoPredictor(model_path=model_path)

# Make prediction
print("\n3. Predicting next hour for Bitcoin...")
result = predictor.predict_next("bitcoin")

if 'error' in result:
    print(f"   Error: {result['error']}")
else:
    print(f"\n   Results:")
    for key, value in result.items():
        print(f"   {key}: {value}")

# Test with threshold
print("\n4. Testing with 60% confidence threshold...")
result_threshold = predictor.predict_with_threshold("bitcoin", confidence_threshold=0.6)
print(f"   Prediction: {result_threshold['prediction']}")
if 'reason' in result_threshold:
    print(f"   Reason: {result_threshold['reason']}")

print("\n" + "=" * 50)
print("Day 9 complete!")