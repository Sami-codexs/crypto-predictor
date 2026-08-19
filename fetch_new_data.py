import subprocess
import time
import sys
import os
from datetime import datetime

print("=" * 60)
print("CryptoMind New Data Collector")
print("Fetches every 60 seconds to avoid CoinGecko rate limits")
print("=" * 60)

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"

success = 0
fail = 0
target = 50  # 50 fetches = ~50 minutes

for i in range(1, target + 1):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{now}] Fetch {i}/{target}...")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/fetch_data.py"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        
        if result.returncode == 0:
            success += 1
            # Extract records stored
            out = result.stdout
            if "records_stored" in out:
                # Find the number
                import re
                m = re.search(r"'records_stored':\s*(\d+)", out)
                if m:
                    print(f"  Stored {m.group(1)} new record(s)")
                else:
                    print(f"  OK")
            else:
                print(f"  OK")
        else:
            fail += 1
            err = result.stderr.strip()[:100] if result.stderr else "Error"
            if "429" in err:
                print(f"  RATE LIMITED - waiting extra time...")
                time.sleep(60)
            else:
                print(f"  FAIL: {err[:80]}")
                
    except Exception as e:
        fail += 1
        print(f"  ERROR: {e}")
    
    # Progress
    if i < target:
        print(f"  Waiting 60s for next fetch...")
        time.sleep(60)

print(f"\n{'='*60}")
print(f"DONE! Success: {success} | Failed: {fail}")
print(f"Total time: ~{target} minutes")
print("Refresh your dashboard (press R) to see new predictions!")
print("=" * 60)