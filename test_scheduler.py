from src.config import setup_logging
from src.scheduler import run_once
import time

setup_logging()

print("Day 3 Test: Scheduler")
print("=" * 40)

print("\n1. Running pipeline once (simulating scheduled job)...")
run_once()

print("\n2. If you want to test continuous scheduling, run:")
print("   python src/scheduler.py")
print("   (Press Ctrl+C to stop)")

print("\n" + "=" * 40)
print("Day 3 complete")