import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data/backtest_5m_precision_rolling_wfa_tp0.018_sl0.012_trailing.json'

print("Waiting for trailing backtest to finish...")
while not os.path.exists(file_path):
    time.sleep(5)

print("Trailing backtest finished! Plotting...")

plt.figure(figsize=(14, 7))

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

trades = data.get('trade_logs', [])

balances = [10000000]
dates = ['2026-03-01']

for t in trades:
    balances.append(t['ending_capital'])
    dates.append(t['exit_dt'][:10])

dt_dates = pd.to_datetime(dates)

plt.plot(dt_dates, balances, label='Best State (TP 1.8 / SL 1.2) + ATR Trailing (Trigger 1.0%, Gap 0.3%)', color='purple', linewidth=2)
plt.scatter(dt_dates, balances, color='orange', marker='o', s=30, zorder=5)

plt.title('KOSPI Lamos WFA - Best State with Trailing Stop (1.0% Trigger, 0.3% Gap)')
plt.xlabel('Date')
plt.ylabel('Account Balance (KRW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('data/wfa_equity_trailing.png')

print("Plotting done.")
