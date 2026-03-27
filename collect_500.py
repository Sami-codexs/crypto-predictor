#!/usr/bin/env python3
"""
Emergency Data Collection - 500 Records
Respects CoinGecko API rate limits (free tier: ~30 calls/minute)
"""

import sys
sys.path.insert(0, 'src')

from fetch_data import CoinGeckoFetcher
from config import setup_logging
from database import CryptoDatabase
import time
from datetime import datetime

setup_logging()

print("=" * 60)
print("🚀 EMERGENCY DATA COLLECTION")
print("=" * 60)
print()
print("Target: 500 records")
print("API Rate Limit: ~20 seconds between calls (safe for free tier)")
print("Estimated Time: ~2.5 hours")
print()
print("You can leave this running overnight!")
print("Press Ctrl+C to stop (progress is saved)")
print("=" * 60)

fetcher = CoinGeckoFetcher()
target = 500
count = 0
consecutive_errors = 0
max_errors = 5

# Check current count
try:
    db = CryptoDatabase()
    existing = db.get_recent_prices("bitcoin", hours=9999)
    count = len(existing)
    print(f"\n📊 Existing records: {count}")
except:
    count = 0

print(f"🎯 Target: {target} records")
print(f"⏳ Remaining: {max(0, target - count)}")
print()

while count < target:
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching record {count + 1}/{target}...")
        
        result = fetcher.fetch_and_store(["bitcoin"])
        
        if result['status'] == 'success' and result['records_stored'] > 0:
            count += result['records_stored']
            consecutive_errors = 0
            print(f"  ✅ Stored! Total: {count}/{target}")
            
            # Progress bar
            pct = (count / target) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"  Progress: |{bar}| {pct:.1f}%")
            
        else:
            print(f"  ⚠️  No data stored. Status: {result.get('status', 'unknown')}")
            consecutive_errors += 1
        
        # Wait 20 seconds between calls (safe for free tier)
        # CoinGecko free: ~30 calls/minute = 1 call per 2 seconds, 
        # but we use 20s to be extra safe and avoid IP bans
        if count < target:
            wait_time = 20
            print(f"  ⏳ Waiting {wait_time}s...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")
        print(f"📊 Final count: {count}/{target}")
        break
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        consecutive_errors += 1
        
        if consecutive_errors >= max_errors:
            print(f"\n🚨 {max_errors} consecutive errors. Stopping.")
            print("Check your internet connection or API limits.")
            break
            
        print("  ⏳ Waiting 60s before retry...")
        time.sleep(60)

print("\n" + "=" * 60)
if count >= target:
    print("✅ TARGET REACHED!")
else:
    print("⚠️  Collection incomplete")
print(f"📊 Total records: {count}/{target}")
print("=" * 60)

# Verify
try:
    db = CryptoDatabase()
    df = db.get_recent_prices("bitcoin", hours=9999)
    print(f"\n✅ Verified: {len(df)} records in database")
    if len(df) > 0:
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
except Exception as e:
    print(f"\n⚠️  Could not verify: {e}")