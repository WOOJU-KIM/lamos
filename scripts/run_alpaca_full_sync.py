import os
import sys
import json
import time
import urllib.request
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import DATA_DIR

DB_PATH = DATA_DIR / "market_data.db"
SYMBOLS = ["SOXL", "SOXS", "TQQQ", "SQQQ", "SOXX", "NVDA", "QQQ", "VIXY", "IEF"]
TIMEFRAMES = [("15m", "15Min"), ("5m", "5Min"), ("60m", "1Hour")]

def fetch_alpaca_history(symbol: str, alpaca_tf: str, start_dt: str, end_dt: str):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API Key 누락! (.env 파일 확인)")
        return pd.DataFrame()
        
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    all_bars = []
    page_token = None
    
    while True:
        url = f"{base_url}?symbols={symbol}&timeframe={alpaca_tf}&start={start_dt}T00:00:00Z&end={end_dt}T23:59:59Z&feed=iex&adjustment=split&limit=10000"
        if page_token:
            url += f"&page_token={page_token}"
            
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json"
        })
        
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
            if "bars" in data and symbol in data["bars"]:
                all_bars.extend(data["bars"][symbol])
                
            page_token = data.get("next_page_token")
            if not page_token:
                break
            time.sleep(0.1) # Rate limit 방지
        except Exception as e:
            print(f"❌ [API 에러] {symbol} {alpaca_tf}: {e}")
            break
            
    if not all_bars:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_bars)
    df = df.rename(columns={'t': 'datetime', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
    # Convert UTC to NYT (US/Eastern) for consistency
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].drop_duplicates(subset=['datetime']).sort_values('datetime')
    return df

def run_alpaca_full_sync():
    print("=" * 80)
    print("🚀 [Lumos 데이터 완전 동기화 (Self-Healing)]")
    print("   • 소스: Alpaca IEX (adjustment=split 적용)")
    print("   • 범위: 최대 기간 (2020-01-01 ~ 오늘)")
    print("=" * 80)
    
    start_dt = "2020-01-01"
    end_dt = datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    total_upserted = 0
    for sym in SYMBOLS:
        for db_tf, alp_tf in TIMEFRAMES:
            # SOXX는 60m만 필요
            # if sym == "SOXX" and db_tf != "60m": continue
            # if sym != "SOXX" and db_tf == "60m": continue
            
            print(f"⏳ {sym} [{db_tf}] 과거 5년치 수정주가 데이터 다운로드 중...")
            df = fetch_alpaca_history(sym, alp_tf, start_dt, end_dt)
            
            if df.empty:
                print(f"   ⚠️ {sym} [{db_tf}] 데이터 없음. 스킵.")
                continue
                
            df['symbol'] = sym
            df['timeframe'] = db_tf
            
            # DB 삽입 (UPSERT)
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO ohlcv (symbol, timeframe, datetime, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, timeframe, datetime) 
                    DO UPDATE SET 
                        open=excluded.open,
                        high=excluded.high,
                        low=excluded.low,
                        close=excluded.close,
                        volume=excluded.volume;
                """, (row['symbol'], row['timeframe'], row['datetime'], 
                      row['Open'], row['High'], row['Low'], row['Close'], row['Volume']))
                
            conn.commit()
            print(f"   ✅ {sym} [{db_tf}]: {len(df):,}개 캔들 DB 덮어쓰기 완료!")
            total_upserted += len(df)
            
    conn.close()
    print("=" * 80)
    print(f"🎉 모든 심볼 데이터 동기화 완료! (총 {total_upserted:,}개 레코드 무결성 확보)")
    print("=" * 80)

if __name__ == "__main__":
    run_alpaca_full_sync()
