from src.config import setup_logging
from src.validation import DataValidator
from src.fetch_data import CoinGeckoFetcher
import pandas as pd

setup_logging()

print("Day 4 Test: Data Validation & Error Recovery")
print("=" * 50)

validator = DataValidator()
fetcher = CoinGeckoFetcher()

# Test 1: Price validation
print("\n1. Testing price validation...")
test_cases = [
    (50000, None, "Normal price"),
    (None, None, "None price"),
    (-100, None, "Negative price"),
    (50000, 40000, "25% jump (suspicious)"),
    (50000, 49000, "2% change (normal)"),
]

for price, prev, desc in test_cases:
    is_valid, msg = validator.validate_price(price, prev)
    status = "✅" if is_valid else "❌"
    print(f"   {status} {desc}: {msg}")

# Test 2: DataFrame validation
print("\n2. Testing DataFrame validation...")
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=5, freq='H'),
    'price_usd': [45000, 46000, None, 1000000, 47000]
})

is_valid, cleaned, errors = validator.validate_dataframe(df)
print(f"   Original: {len(df)} rows, Cleaned: {len(cleaned)} rows")
print(f"   Errors found: {len(errors)}")
for err in errors[:3]:
    print(f"   - {err}")

# Test 3: Fetch with retry
print("\n3. Testing fetch with retry logic...")
result = fetcher.fetch_and_store(["bitcoin"])
print(f"   Result: {result}")

print("\n" + "=" * 50)
print("Day 4 complete")