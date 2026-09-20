import urllib.request
import json
import sqlite3
import pandas as pd
import time
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "market_data.db"
BEARER_TOKEN = "rV1TQcaxLSMZ-GMbsZRoAmJXxJlGBzFGmpCtq_ohhYo2V1-WXKOtDbzjbEDIEz7EUyt4PbNjhidsLjVHmq-5ww"
CANO = "56911332"

def fetch_kiwoom_sector_history(inds_cd, tic_scope):
    print(f"[{inds_cd}] Fetching {tic_scope}m from Kiwoom REST API (ka20005)...")
    url = "https://api.kiwoom.com/api/dostk/chart"
    all_data = []
    
    headers = {
        'Content-Type': 'application/json;charset=UTF-8',
        'authorization': f'Bearer {BEARER_TOKEN}',
        'api-id': 'ka20005',
        'cont-yn': 'N',
        'next-key': ''
    }
    
    body = {
        'cano': CANO,
        'acnt_prdt_cd': '01',
        'inds_cd': inds_cd,
        'tic_scope': tic_scope
    }
    
    page = 1
    while True:
        req = urllib.request.Request(url, data=json.dumps(body).encode('utf-8'), headers=headers, method='POST')
        try:
            res = urllib.request.urlopen(req)
            cont_yn = res.headers.get('cont-yn', 'N')
            next_key = res.headers.get('next-key', '')
            
            res_data = json.loads(res.read().decode('utf-8'))
            
            if 'inds_min_pole_qry' not in res_data:
                break
                
            items = res_data['inds_min_pole_qry']
            if not items:
                break
                
            all_data.extend(items)
            print(f"  -> Page {page}: fetched {len(items)} rows. NextKey: {next_key}")
            
            if cont_yn == 'Y' and next_key:
                headers['cont-yn'] = 'Y'
                headers['next-key'] = next_key
                page += 1
                time.sleep(1.0)
            else:
                break
                
        except Exception as e:
            print(f"  -> Exception: {e}")
            break
            
    if not all_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_data)
    
    for col in ['cur_prc', 'open_pric', 'high_pric', 'low_pric']:
        df[col] = df[col].astype(str).str.replace('+', '').str.replace('-', '').astype(float)
        
    # Kiwoom sector index prices are stored as * 100 integers usually?
    # e.g., cur_prc = 4356 means 43.56 ?
    # Let's check VKOSPI current value, it's around 15-20 usually. 
    # Ah, 4356 -> 43.56? Wait, 2026 VKOSPI might be different. Let's just store the raw or divide by 100.
    # Usually index values from Kiwoom TR are multiplied by 100. Let's divide by 100 to get the real index value.
    for col in ['cur_prc', 'open_pric', 'high_pric', 'low_pric']:
        df[col] = df[col] / 100.0
        
    df['trde_qty'] = df['trde_qty'].astype(float)
    df['datetime'] = pd.to_datetime(df['cntr_tm'], format='%Y%m%d%H%M%S')
    
    df.rename(columns={
        'open_pric': 'open',
        'high_pric': 'high',
        'low_pric': 'low',
        'cur_prc': 'close',
        'trde_qty': 'volume'
    }, inplace=True)
    
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    # Map back to our internal symbol name so it matches config.MACRO_TICKER_3
    df['symbol'] = config.MACRO_TICKER_3 # "^VKOSPI"
    df['timeframe'] = f"{tic_scope}m"
    
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    
    # VKOSPI is sector code "603"
    sector_code = "603"
    
    for timeframe in ['60', '15', '5']:
        df = fetch_kiwoom_sector_history(sector_code, timeframe)
        if not df.empty:
            df['datetime'] = df['datetime'].astype(str)
            try:
                # delete existing yfinance data for VKOSPI if any
                cursor = conn.cursor()
                cursor.execute("DELETE FROM market_candles WHERE symbol=? AND timeframe=?", (config.MACRO_TICKER_3, f"{timeframe}m"))
                conn.commit()
                
                df.to_sql('market_candles', conn, if_exists='append', index=False, chunksize=1000)
                print(f"  -> Saved {len(df)} rows of {timeframe}m to DB.")
            except Exception as e:
                print(f"DB Error: {e}")
                
    conn.close()
    print("VKOSPI fetching via Kiwoom REST completed!")
