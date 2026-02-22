import sys
sys.path.insert(0, 'src')

from scheduler import run_once
from config import setup_logging
import time

setup_logging()

print("Collecting data every 20 minutes. Press Ctrl+C to stop.")
print("=" * 50)

count = 0
while True:
    try:
        count += 1
        print(f"\nFetch #{count}")
        run_once()
        print("Sleeping 20 minutes...")
        time.sleep(1200)  # 20 minutes
    except KeyboardInterrupt:
        print("\nStopped.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(60)  # Wait 1 min on error