import requests
import json

BASE_URL = "http://localhost:8000"

print("Day 16 Test: Batch & Advanced Features")
print("=" * 50)

# Test 1: Batch prediction
print("\n1. Batch prediction (Bitcoin + Ethereum)...")
try:
    r = requests.post(
        f"{BASE_URL}/predict/batch",
        json={
            "coins": ["bitcoin", "ethereum"],
            "confidence_threshold": 0.55
        },
        timeout=60
    )
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        summary = data.get('summary', {})
        print(f"   Summary: {summary}")
        
        for pred in data.get('predictions', []):
            coin = pred.get('coin', 'unknown')
            prediction = pred.get('prediction', 'error')
            conf = pred.get('confidence', 0)
            print(f"   {coin}: {prediction} (conf: {conf:.2f})")
    else:
        print(f"   Error: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Invalid batch (too many coins)
print("\n2. Batch with 15 coins (should fail)...")
try:
    r = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"coins": ["bitcoin"] * 15, "confidence_threshold": 0.6},
        timeout=10
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Model comparison
print("\n3. Model comparison...")
try:
    r = requests.get(f"{BASE_URL}/models/compare", timeout=10)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Total models: {data.get('total_models')}")
        print(f"   Best model: {data.get('best_model')}")
        for model in data.get('models', [])[:3]:
            print(f"   - {model['name']}: {model.get('accuracy', 'N/A')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Prediction history
print("\n4. Prediction history...")
try:
    r = requests.get(f"{BASE_URL}/predictions/history?coin=bitcoin&hours=24", timeout=10)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Note: {r.json().get('note')}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("Day 16 complete!")