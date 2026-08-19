"""
Historical Data Fetcher for CryptoMind.
Fetches 365 days of Bitcoin price data from CoinGecko in ONE API call.
"""
import requests
import pandas as pd
import sqlite3
import time
import sys
import os
from datetime import datetime

DB_PATH = os.path.join("src", "data", "crypto.db")

def fetch_historical_data(coin_id="bitcoin", vs_currency="usd", days=365):
    """Fetch historical market data from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
    }
    
    print(f"Fetching {days} days of historical data for {coin_id}...")
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    # Extract prices and volumes
    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])
    market_caps = data.get("market_caps", [])
    
    if not prices:
        print("No price data returned!")
        return pd.DataFrame()
    
    # Build DataFrame
    df = pd.DataFrame(prices, columns=["timestamp", "price_usd"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Add volume
    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume_24h"])
        vol_df["timestamp"] = pd.to_datetime(vol_df["timestamp"], unit="ms")
        df = df.merge(vol_df, on="timestamp", how="left")
    else:
        df["volume_24h"] = 0.0
    
    # Add market cap
    if market_caps:
        cap_df = pd.DataFrame(market_caps, columns=["timestamp", "market_cap"])
        cap_df["timestamp"] = pd.to_datetime(cap_df["timestamp"], unit="ms")
        df = df.merge(cap_df, on="timestamp", how="left")
    else:
        df["market_cap"] = 0.0
    
    # Add coin_id
    df["coin_id"] = coin_id
    
    print(f"Retrieved {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df

def save_to_db(df, db_path=DB_PATH):
    """Save DataFrame to SQLite, avoiding duplicates."""
    if df.empty:
        return 0
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Check existing timestamps for this coin
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
    if cursor.fetchone():
        existing = pd.read_sql("SELECT DISTINCT timestamp FROM prices WHERE coin_id = ?", conn, params=(df["coin_id"].iloc[0],))
        existing_timestamps = set(existing["timestamp"].tolist())
    else:
        existing_timestamps = set()
    
    # Filter out duplicates
    df["timestamp_str"] = df["timestamp"].astype(str)
    new_rows = df[~df["timestamp_str"].isin(existing_timestamps)]
    
    if len(new_rows) == 0:
        print("No new records to save (all timestamps already exist).")
        conn.close()
        return 0
    
    # Save new rows
    save_df = new_rows[["timestamp", "coin_id", "price_usd", "volume_24h", "market_cap"]].copy()
    save_df.to_sql("prices", conn, if_exists="append", index=False)
    
    conn.commit()
    conn.close()
    print(f"Saved {len(save_df)} new records to database.")
    return len(save_df)

def main():
    print("=" * 60)
    print("CryptoMind Historical Data Fetcher")
    print("=" * 60)
    
    try:
        df = fetch_historical_data(days=365)
        if not df.empty:
            count = save_to_db(df)
            print(f"\nDone! Stored {count} new records.")
            print(f"Database now has data from {df['timestamp'].min()} to {df['timestamp'].max()}")
        else:
            print("No data fetched.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("ERROR: CoinGecko rate limit hit. Wait 1-2 minutes and try again.")
        else:
            print(f"HTTP Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()