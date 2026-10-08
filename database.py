import sqlite3
import os
from config import DB_PATH

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 거래 내역 테이블 (Trade History)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        ticker TEXT NOT NULL,
        trade_type TEXT NOT NULL, -- 'BUY', 'SELL_TP' (익절), 'SELL_SL' (손절)
        price REAL NOT NULL,
        shares INTEGER NOT NULL,
        total_amount REAL NOT NULL,
        realized_pnl REAL, -- 매도 시 실현 손익
        return_rate REAL   -- 매도 시 수익률
    )
    ''')
    
    # 매수 후보군 (Daily Candidates)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_candidates (
        date DATE NOT NULL,
        rank INTEGER NOT NULL,
        ticker TEXT NOT NULL,
        z_score REAL,
        PRIMARY KEY (date, rank)
    )
    ''')
    
    # 현재 포트폴리오 (Current Portfolio - 키움 잔고와 동기화용)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        ticker TEXT PRIMARY KEY,
        buy_price REAL NOT NULL,
        shares INTEGER NOT NULL,
        buy_date DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ 데이터베이스 초기화 완료 (quant_system.db)")

def log_trade(ticker, trade_type, price, shares, realized_pnl=None, return_rate=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total_amount = price * shares
    cursor.execute('''
    INSERT INTO trade_history (ticker, trade_type, price, shares, total_amount, realized_pnl, return_rate)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, trade_type, price, shares, total_amount, realized_pnl, return_rate))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
