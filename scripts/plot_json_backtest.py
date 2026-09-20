import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def plot_monthly_json():
    file_path = Path("data/backtest_5m_precision_rolling_wfa.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    monthly = data.get("monthly", {})
    if not monthly: return
    
    dates = []
    caps = []
    for m, stats in monthly.items():
        dates.append(m)
        caps.append(stats['ending_capital'])
        
    plt.figure(figsize=(10, 5))
    plt.plot(dates, caps, marker='o', color='red')
    plt.title('Monthly Portfolio Equity (KRW)', fontsize=14)
    plt.xlabel('Month')
    plt.ylabel('Capital (KRW)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = Path("data/backtest_chart.png")
    plt.savefig(out_path, dpi=150)

if __name__ == '__main__':
    plot_monthly_json()
