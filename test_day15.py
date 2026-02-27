import requests

BASE_URL = "http://localhost:8000"

print("Day 15 Test: Performance Endpoint")
print("=" * 50)

# Test 1: Get performance metrics
print("\n1. Getting performance metrics (7 days)...")
try:
    r = requests.get(f"{BASE_URL}/performance?coin=bitcoin&days=7", timeout=60)
    print(f"   Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        print(f"\n   Model: {data.get('model_name')}")
        print(f"   Accuracy: {data.get('accuracy'):.1%}")
        print(f"   Win Rate: {data.get('win_rate'):.1%}")
        print(f"   Total Return: {data.get('total_return_pct'):+.2f}%")
        print(f"   Sharpe Ratio: {data.get('sharpe_ratio'):.2f}")
        print(f"   Max Drawdown: {data.get('max_drawdown_pct'):.2f}%")
        print(f"   Trades: {data.get('total_trades')}")
    else:
        print(f"   Error: {r.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Invalid days parameter
print("\n2. Invalid days (should fail)...")
try:
    r = requests.get(f"{BASE_URL}/performance?coin=bitcoin&days=50", timeout=10)
    print(f"   Status: {r.status_code}")
    print(f"   Response: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Short period (1 day)
print("\n3. Short period (1 day)...")
try:
    r = requests.get(f"{BASE_URL}/performance?coin=bitcoin&days=1", timeout=30)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        print(f"   Trades: {r.json().get('total_trades')}")
    else:
        print(f"   Error: {r.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 50)
print("Day 15 complete!")