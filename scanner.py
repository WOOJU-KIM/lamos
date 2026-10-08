import yfinance as yf
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sqlite3
from config import DB_PATH, SP500_UNIVERSE_SIZE, MAX_Z_SCORE, SCANNER_CANDIDATES_COUNT, BLACKLIST_TICKERS

def run_premarket_scan():
    print("Pre-market Scan Started...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=6) # 6년치 받아야 5년치(1260거래일) Z-Score가 계산됨
    
    print("Downloading data...")
    data_unadj = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False, actions=True, threads=True)
    prices_u = data_unadj['Close'].ffill()
    volumes = data_unadj['Volume'].ffill()
    
    divs = data_unadj.get('Dividends')
    if divs is None:
        divs = pd.DataFrame(0, index=prices_u.index, columns=prices_u.columns)
    else:
        divs = divs.fillna(0)
        
    prices_u.index = prices_u.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    volumes.index = volumes.index.tz_localize(None)
    
    print(f"Filtering Top {SP500_UNIVERSE_SIZE} by Dollar Volume...")
    recent_dollar_vol = (prices_u.iloc[-20:] * volumes.iloc[-20:]).mean()
    top_tickers = recent_dollar_vol.sort_values(ascending=False).head(SP500_UNIVERSE_SIZE).index.tolist()
    
    prices_u = prices_u[top_tickers]
    divs = divs[top_tickers]
    
    print("Calculating Z-Scores (Daily)...")
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    
    roll_mean = div_yield.rolling(window=1260, min_periods=252).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=252).std()
    
    daily_z_score = (div_yield - roll_mean) / roll_std
    latest_z = daily_z_score.iloc[-1].dropna()
    
    # Z-Score가 설정값(MAX_Z_SCORE)을 초과하는 것은 특별배당 등 왜곡으로 간주하여 제외
    latest_z = latest_z[latest_z <= MAX_Z_SCORE]
    
    # 영구 제명 종목 (사용자 지정 블랙리스트) config 적용
    latest_z = latest_z[~latest_z.index.isin(BLACKLIST_TICKERS)].sort_values(ascending=False)
    
    candidates = latest_z.head(SCANNER_CANDIDATES_COUNT)
    
    print("Saving to database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    cursor.execute("DELETE FROM daily_candidates WHERE date = ?", (today_str,))
    
    for rank, (tk, z_val) in enumerate(candidates.items(), 1):
        cursor.execute("INSERT INTO daily_candidates (date, rank, ticker, z_score) VALUES (?, ?, ?, ?)",
                       (today_str, rank, tk, z_val))
                       
    conn.commit()
    conn.close()
    print("Pre-market Scan Completed. Candidates saved to DB.")

if __name__ == "__main__":
    run_premarket_scan()
