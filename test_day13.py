import requests
import json

BASE_URL = "http://localhost:8000"

print("Day 13 Test: FastAPI Endpoints")
print("=" * 50)

# Test 1: Root
print("\n1. Testing root endpoint...")
try:
    r = requests.get(f"{BASE_URL}/", timeout=5)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Health
print("\n2. Testing health endpoint...")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"   Status: {r.status_code}")
    data = r.json()
    print(f"   Model loaded: {data.get('model_loaded')}")
    print(f"   Database: {data.get('database_connected')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Predict (if model loaded)
print("\n3. Testing prediction...")
try:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"coin": "bitcoin", "confidence_threshold": 0.6},
        timeout=30
    )
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Prediction: {data.get('prediction')}")
        print(f"   Confidence: {data.get('confidence')}")
        print(f"   Price: ${data.get('current_price')}")
    else:
        print(f"   Error: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("Day 13 test complete!")
print("\nTo run server: python -m uvicorn src.api:app --reload")