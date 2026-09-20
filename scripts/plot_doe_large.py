import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

files = [
    'data/backtest_5m_precision_rolling_wfa_tp0.015_sl0.01.json',
    'data/backtest_5m_precision_rolling_wfa_tp0.018_sl0.012.json',
    'data/backtest_5m_precision_rolling_wfa_tp0.021_sl0.014.json',
    'data/backtest_5m_precision_rolling_wfa_tp0.024_sl0.016.json',
    'data/backtest_5m_precision_rolling_wfa_tp0.027_sl0.018.json',
    'data/backtest_5m_precision_rolling_wfa_tp0.03_sl0.02.json'
]

labels = [
    'TP 1.5% / SL 1.0%',
    'TP 1.8% / SL 1.2%',
    'TP 2.1% / SL 1.4%',
    'TP 2.4% / SL 1.6%',
    'TP 2.7% / SL 1.8%',
    'TP 3.0% / SL 2.0%'
]

print("Waiting for all DoE runs to finish...")
while True:
    all_exist = all(os.path.exists(f) for f in files)
    if all_exist:
        break
    time.sleep(10)

print("All DoE runs finished! Plotting...")

plt.figure(figsize=(16, 9))

for file, label in zip(files, labels):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trades = data.get('trade_logs', [])
    
    balances = [10000000]
    dates = ['2026-03-01']
    
    for t in trades:
        balances.append(t['ending_capital'])
        dates.append(t['exit_dt'][:10])
        
    dt_dates = pd.to_datetime(dates)
    
    plt.plot(dt_dates, balances, label=label, linewidth=2)

plt.title('DoE (Large Swing): KOSPI Lamos WFA 1.5:1 Risk/Reward Scenarios')
plt.xlabel('Date')
plt.ylabel('Account Balance (KRW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('data/doe_large_swing_equity.png')

print("Plotting done.")
