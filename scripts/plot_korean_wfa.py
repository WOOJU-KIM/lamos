import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load OOS trades
with open('data/backtest_5m_precision_rolling_wfa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

trades = data.get('trade_logs', [])

# 2. Fetch price data for 122630 (5m)
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql_query("SELECT datetime, close FROM market_candles WHERE symbol='122630' AND timeframe='5m'", conn)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)
df.sort_index(inplace=True)

# 3. Filter OOS period (2026-03-01 to end)
df = df[df.index >= '2026-03-01']

# 4. Plot
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['close'], label='KODEX Leverage (122630)', color='black', linewidth=1, alpha=0.7)

# 5. Scatter Trades
buy_times = []
buy_prices = []
sell_times = []
sell_prices = []

for t in trades:
    # 2026-03-05 10:15:00
    try:
        et = pd.to_datetime(t['entry_time'])
        xt = pd.to_datetime(t['exit_time'])
        if et >= pd.to_datetime('2026-03-01'):
            buy_times.append(et)
            buy_prices.append(t['entry_price'])
            sell_times.append(xt)
            sell_prices.append(t['exit_price'])
    except:
        pass

plt.scatter(buy_times, buy_prices, marker='^', color='green', s=100, label='Long Entry', zorder=5)
plt.scatter(sell_times, sell_prices, marker='v', color='red', s=100, label='Long Exit', zorder=5)

plt.title('KOSPI True Lamos WFA - Entry/Exit Points on KODEX Leverage (Out of Sample)')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('data/wfa_points_chart.png')
