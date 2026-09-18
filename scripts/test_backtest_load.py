import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Load all 15m data from SQLite
conn = sqlite3.connect('data/market_data.db')
tqqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' ORDER BY datetime", conn)
sqqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='SQQQ' AND timeframe='15m' ORDER BY datetime", conn)
nvda_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
qqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
vix_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
tnx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)
soxx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='SOXX' AND timeframe='15m' ORDER BY datetime", conn)

print(f"Data loaded: TQQQ={len(tqqq_df)}, SQQQ={len(sqqq_df)}, NVDA={len(nvda_df)}, QQQ={len(qqq_df)}, VIX={len(vix_df)}, TNX={len(tnx_df)}, SOXX={len(soxx_df)}")
