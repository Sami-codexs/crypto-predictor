import requests
import json
import time

BASE_URL = "http://localhost:8000"

print("Day 14 Test: Validation & Error Handling")
print("=" * 50)

# Test 1: Valid prediction
print("\n1. Valid prediction request...")
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
    else:
        print(f"   Error: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Invalid coin
print("\n2. Invalid coin (should fail)...")
try:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"coin": "dogecoin", "confidence_threshold": 0.6},
        timeout=10
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Invalid threshold
print("\n3. Invalid threshold > 1.0 (should fail)...")
try:
    r = requests.post(
        f"{BASE_URL}/predict",
        json={"coin": "bitcoin", "confidence_threshold": 1.5},
        timeout=10
    )
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Rate limiting (10 quick requests)
print("\n4. Rate limiting test (11 requests)...")
success_count = 0
rate_limited = False
for i in range(11):
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            success_count += 1
        elif r.status_code == 429:
            rate_limited = True
            print(f"   Request {i+1}: Rate limited (429)")
            break
    except Exception as e:
        print(f"   Request {i+1}: Error - {e}")
        break
    time.sleep(0.1)

print(f"   Successful: {success_count}, Rate limited: {rate_limited}")

# Test 5: History with validation
print("\n5. History endpoint validation...")
try:
    # Valid
    r = requests.get(f"{BASE_URL}/history?coin=bitcoin&hours=24", timeout=10)
    print(f"   Valid: Status {r.status_code}")
    
    # Invalid hours
    r = requests.get(f"{BASE_URL}/history?coin=bitcoin&hours=999", timeout=10)
    print(f"   Invalid hours: Status {r.status_code}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("Day 14 complete!")