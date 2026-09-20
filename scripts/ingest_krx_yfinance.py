import sqlite3
import yfinance as yf
import pandas as pd
import datetime
from pathlib import Path
import sys
import os

# Add parent dir to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "market_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_candles (
            symbol TEXT,
            timeframe TEXT,
            datetime DATETIME,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, datetime)
        )
    """)
    conn.commit()
    return conn

def download_and_save(conn, symbol, yf_symbol, timeframe, period="max"):
    print(f"[{symbol}] Downloading {timeframe} data from yfinance (period={period})...")
    try:
        df = yf.download(yf_symbol, period=period, interval=timeframe, progress=False)
        if df.empty:
            print(f"  -> No data found for {yf_symbol}")
            return
            
        df = df.reset_index()
        # Rename Date or Datetime column to 'datetime'
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'datetime'}, inplace=True)
        elif 'Date' in df.columns:
            df.rename(columns={'Date': 'datetime'}, inplace=True)
            
        # Clean columns if it's MultiIndex (yf.download sometimes returns MultiIndex in newer versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Ensure correct case for OHLCV
        df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        
        # Keep only needed columns
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[cols].copy()
        
        # Add symbol and timeframe
        df['symbol'] = symbol
        df['timeframe'] = timeframe
        
        # Convert datetime to string for SQLite
        df['datetime'] = df['datetime'].astype(str)
        
        # Save to DB
        df.to_sql('market_candles', conn, if_exists='append', index=False, method='multi')
        print(f"  -> Successfully saved {len(df)} rows for {symbol} ({timeframe})")
    except sqlite3.IntegrityError:
        # Ignore duplicates
        print(f"  -> Data already exists for {symbol} ({timeframe})")
    except Exception as e:
        print(f"  -> Error: {e}")

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    conn = init_db()
    
    # Map of our internal symbols to yfinance symbols
    # Korean ETFs/Stocks need '.KS' suffix (or '.KQ' for KOSDAQ)
    # Indices like ^VKOSPI are already compatible
    yf_mapping = {}
    for sym in config.ALL_TICKERS:
        if sym.startswith("^"):
            yf_mapping[sym] = sym
        else:
            yf_mapping[sym] = f"{sym}.KS"
            
    print(f"Target Tickers: {yf_mapping}")
    
    # Download configurations
    # yfinance limits: 5m, 15m, 60m -> max 60 days
    # 1d -> max (since 1990s)
    
    for sym, yf_sym in yf_mapping.items():
        download_and_save(conn, sym, yf_sym, "1d", period="max")
        download_and_save(conn, sym, yf_sym, "60m", period="730d") # yfinance max for 60m is 730d!
        download_and_save(conn, sym, yf_sym, "15m", period="60d")
        download_and_save(conn, sym, yf_sym, "5m", period="60d")
        
    # Deduplicate DB (just in case)
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM market_candles
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM market_candles
            GROUP BY symbol, timeframe, datetime
        )
    """)
    conn.commit()
    conn.close()
    
    print("\n✅ All available maximum data downloaded and saved to data/market_data.db!")
