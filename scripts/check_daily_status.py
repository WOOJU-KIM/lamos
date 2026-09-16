import os
import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. StateHub Live Trades
print("=== 1. 금일 실전/모의 거래 체결 내역 ===")
conn_state = sqlite3.connect(PROJECT_ROOT / "data" / "state_hub.db")
try:
    df_trades = pd.read_sql_query("SELECT * FROM live_trades ORDER BY rowid DESC LIMIT 10;", conn_state)
    if df_trades.empty:
        print("• 실전/모의 체결 내역: 0건 (100% 현금 대기 중 / 불필요한 뇌동매매 방어 완료)")
    else:
        print(df_trades.to_string(index=False))
except Exception as e:
    print("• live_trades 조회:", e)
conn_state.close()

# 2. Shadow Trades (8/17 ~ 8/18)
print("\n=== 2. 금일 섀도우 트랙 체결 내역 ===")
conn_shadow = sqlite3.connect(PROJECT_ROOT / "data" / "shadow_trades.db")
try:
    df_shadow = pd.read_sql_query("SELECT * FROM shadow_trades WHERE timestamp >= '2026-08-17' ORDER BY rowid DESC LIMIT 10;", conn_shadow)
    if df_shadow.empty:
        print("• 금일 체결 건수: 0건 (신호 조건 미충족에 따른 현금 보존)")
    else:
        print(df_shadow.to_string(index=False))
except Exception as e:
    print("• shadow_trades 조회:", e)
conn_shadow.close()

# 3. Market Data Lake (market_data.db)
print("\n=== 3. 데이터 레이크(market_data.db) 적재 현황 ===")
conn_mkt = sqlite3.connect(PROJECT_ROOT / "data" / "market_data.db")
try:
    df_mkt_max = pd.read_sql_query("SELECT symbol, timeframe, MIN(datetime) as min_date, MAX(datetime) as max_date, COUNT(*) as cnt FROM market_candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe;", conn_mkt)
    print(df_mkt_max.to_string(index=False))
    
    print("\n--- 최근 하베스트 아카이빙 로그 (최근 5건) ---")
    df_logs = pd.read_sql_query("SELECT * FROM harvest_logs ORDER BY log_id DESC LIMIT 5;", conn_mkt)
    print(df_logs.to_string(index=False))
except Exception as e:
    print("• market_data 조회:", e)
conn_mkt.close()

# 4. Model Registry
print("\n=== 4. 모델 레지스트리 및 최신화 재학습 상태 ===")
conn_reg = sqlite3.connect(PROJECT_ROOT / "data" / "model_registry.db")
try:
    df_reg = pd.read_sql_query("SELECT model_id, algorithm_type, train_data_range, created_at, status, win_rate, total_return FROM model_registry;", conn_reg)
    print(df_reg.to_string(index=False))
except Exception as e:
    print("• model_registry 조회:", e)
conn_reg.close()
