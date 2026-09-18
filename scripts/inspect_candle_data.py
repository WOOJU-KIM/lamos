import sqlite3
import pandas as pd

conn = sqlite3.connect('data/market_data.db')
df_15m = pd.read_sql_query("SELECT symbol, timeframe, COUNT(*) as cnt, MIN(datetime) as min_dt, MAX(datetime) as max_dt FROM market_candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe", conn)
print(df_15m.to_string())

df_tqqq = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' ORDER BY datetime", conn)
print('TQQQ 15m rows:', len(df_tqqq))
df_tqqq['date'] = df_tqqq['datetime'].str.slice(0, 10)
unique_dates = df_tqqq['date'].unique()
print('Total unique trading days:', len(unique_dates))
print('First 5 days:', unique_dates[:5])
print('Last 5 days:', unique_dates[-5:])
