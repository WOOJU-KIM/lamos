import sqlite3
import yfinance as yf
from database import DB_PATH

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT rank, ticker, z_score FROM daily_candidates ORDER BY rank ASC LIMIT 20")
rows = cursor.fetchall()
conn.close()

print("\n[Top 10 매수 후보 현재 배당수익률 확인]")
for rank, tk, z in rows:
    try:
        info = yf.Ticker(tk).info
        div_yield = info.get('dividendYield', 0)
        # dividendYield는 0.045 (4.5%) 같은 소수점으로 나옴
        if div_yield:
            div_pct = div_yield
        else:
            # trailingAnnualDividendYield 확인
            div_yield = info.get('trailingAnnualDividendYield', 0)
            div_pct = div_yield if div_yield else 0.0
            
        print(f"{rank}위: {tk} | 배당수익률: {div_pct:.2f}% (Z-Score: {z:.2f})")
    except Exception as e:
        print(f"{rank}위: {tk} | 배당수익률: 조회 실패 (Z-Score: {z:.2f})")
