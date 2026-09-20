import urllib.request
import json
import sqlite3
import pandas as pd
import time
import sys
import os
from pathlib import Path

# Add parent dir to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "market_data.db"

# We use the token from config if exists, otherwise the mock token for test (since the user already used this token in previous steps)
# In production, this should fetch a valid token.
BEARER_TOKEN = "rV1TQcaxLSMZ-GMbsZRoAmJXxJlGBzFGmpCtq_ohhYo2V1-WXKOtDbzjbEDIEz7EUyt4PbNjhidsLjVHmq-5ww"
CANO = "56911332"

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

def fetch_kiwoom_history(stk_cd, tic_scope):
    print(f"[{stk_cd}] Fetching {tic_scope}m from Kiwoom REST API...")
    url = "https://api.kiwoom.com/api/dostk/chart"
    all_data = []
    
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'authorization': f'Bearer {BEARER_TOKEN}',
        'api-id': 'ka10080',
        'cont-yn': 'N',
        'next-key': ''
    }
    
    body = {
        'cano': CANO,
        'acnt_prdt_cd': '01',
        'stk_cd': stk_cd,
        'tic_scope': tic_scope,
        'upd_stkpc_tp': '1'
    }
    
    page = 1
    while True:
        req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
        try:
            res = urllib.request.urlopen(req)
            cont_yn = res.headers.get('cont-yn', 'N')
            next_key = res.headers.get('next-key', '')
            
            res_data = json.loads(res.read().decode('utf-8'))
            
            if 'stk_min_pole_chart_qry' not in res_data:
                print(f"Error or end of data: {res_data}")
                break
                
            items = res_data['stk_min_pole_chart_qry']
            if not items:
                break
                
            all_data.extend(items)
            print(f"  -> Page {page}: fetched {len(items)} rows. NextKey: {next_key}")
            
            if cont_yn == 'Y' and next_key:
                headers['cont-yn'] = 'Y'
                headers['next-key'] = next_key
                page += 1
                time.sleep(1.0) # Rate limit
            else:
                break
                
        except Exception as e:
            print(f"  -> Exception: {e}")
            break
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    # columns: cur_prc, trde_qty, cntr_tm, open_pric, high_pric, low_pric
    
    # Parse prices (remove + or - signs)
    for col in ['cur_prc', 'open_pric', 'high_pric', 'low_pric']:
        df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)
        
    df['trde_qty'] = df['trde_qty'].astype(float)
    
    # cntr_tm format: YYYYMMDDHHMMSS (e.g., 20260918153000)
    df['datetime'] = pd.to_datetime(df['cntr_tm'], format='%Y%m%d%H%M%S')
    
    df.rename(columns={
        'open_pric': 'open',
        'high_pric': 'high',
        'low_pric': 'low',
        'cur_prc': 'close',
        'trde_qty': 'volume'
    }, inplace=True)
    
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['symbol'] = stk_cd
    df['timeframe'] = f"{tic_scope}m"
    
    # Sort chronological
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

def save_to_db(df, conn):
    if df.empty:
        return
    df['datetime'] = df['datetime'].astype(str)
    try:
        df.to_sql('market_candles', conn, if_exists='append', index=False, chunksize=1000)
        print(f"  -> Saved {len(df)} rows to DB.")
    except sqlite3.IntegrityError:
        pass
    except Exception as e:
        print(f"  -> DB Save Error: {e}")

if __name__ == "__main__":
    conn = init_db()
    
    # Stocks
    stock_tickers = [
        config.TICKER_LONG,    # 122630
        config.TICKER_SHORT,   # 252670
        config.TICKER_TREND,   # 069500
        config.MACRO_TICKER_2, # 000660
        config.MACRO_TICKER_4  # 114260
    ]
    stock_tickers = list(set(stock_tickers)) # remove duplicates (069500 is there twice)
    
    for sym in stock_tickers:
        for timeframe in ['60', '15', '5']:
            df = fetch_kiwoom_history(sym, timeframe)
            save_to_db(df, conn)
            
    # For ^VKOSPI, we use yfinance (since Kiwoom stock endpoint doesn't support indices natively via ka10080)
    print("\nFetching ^VKOSPI from yfinance...")
    vkospi = config.MACRO_TICKER_3
    try:
        for tf, period in [('60m', '730d'), ('15m', '60d'), ('5m', '60d')]:
            print(f"  -> yfinance {tf} (period={period})")
            df_yf = yf.download(vkospi, period=period, interval=tf, progress=False)
            if not df_yf.empty:
                df_yf = df_yf.reset_index()
                date_col = 'Datetime' if 'Datetime' in df_yf.columns else 'Date'
                df_yf.rename(columns={date_col: 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                if isinstance(df_yf.columns, pd.MultiIndex): df_yf.columns = df_yf.columns.get_level_values(0)
                
                df_yf = df_yf[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
                df_yf['symbol'] = vkospi
                df_yf['timeframe'] = tf
                save_to_db(df_yf, conn)
    except Exception as e:
        print(f"  -> yfinance error: {e}")

    # Cleanup duplicates
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
    print("\nAll fetching completed and saved to DB!")
