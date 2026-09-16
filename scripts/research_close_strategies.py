import sqlite3
import sys
import pandas as pd
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def test_cross_asset_at_close():
    conn = sqlite3.connect('data/market_data.db')
    
    # Load 5m SOXL, QQQ, NVDA, ^VIX
    query = """
        SELECT symbol, datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE timeframe='5m' AND symbol IN ('SOXL', 'QQQ', 'NVDA', '^VIX')
        ORDER BY datetime ASC
    """
    df = pd.read_sql_query(query, conn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    days = sorted(df['date'].unique())
    results = []

    for d in days:
        day_data = df[df['date'] == d]
        
        soxl_5 = day_data[day_data['symbol'] == 'SOXL']
        qqq_5 = day_data[day_data['symbol'] == 'QQQ']
        nvda_5 = day_data[day_data['symbol'] == 'NVDA']
        vix_5 = day_data[day_data['symbol'] == '^VIX']

        if soxl_5.empty or qqq_5.empty or nvda_5.empty:
            continue

        p_1500_soxl = soxl_5[soxl_5['time'] == '15:00']
        p_1550_soxl = soxl_5[soxl_5['time'] == '15:50']
        p_1430_soxl = soxl_5[soxl_5['time'] == '14:30']

        p_1500_qqq = qqq_5[qqq_5['time'] == '15:00']
        p_1430_qqq = qqq_5[qqq_5['time'] == '14:30']

        p_1500_nvda = nvda_5[nvda_5['time'] == '15:00']
        p_1430_nvda = nvda_5[nvda_5['time'] == '14:30']

        if p_1500_soxl.empty or p_1550_soxl.empty or p_1430_soxl.empty or p_1500_qqq.empty or p_1430_qqq.empty or p_1500_nvda.empty or p_1430_nvda.empty:
            continue

        s1500 = p_1500_soxl.iloc[0]['close']
        s1550 = p_1550_soxl.iloc[0]['close']
        s1430 = p_1430_soxl.iloc[0]['close']

        q1500 = p_1500_qqq.iloc[0]['close']
        q1430 = p_1430_qqq.iloc[0]['close']

        n1500 = p_1500_nvda.iloc[0]['close']
        n1430 = p_1430_nvda.iloc[0]['close']

        # Return between 14:30 and 15:00
        soxl_pwr_in = (s1500 - s1430) / s1430 * 100
        qqq_pwr_in = (q1500 - q1430) / q1430 * 100
        nvda_pwr_in = (n1500 - n1430) / n1430 * 100

        # Outcome: 15:00 to 15:50 SOXL return
        soxl_out = (s1550 - s1500) / s1500 * 100

        results.append({
            'date': d,
            'soxl_pwr_in': soxl_pwr_in,
            'qqq_pwr_in': qqq_pwr_in,
            'nvda_pwr_in': nvda_pwr_in,
            'soxl_out': soxl_out
        })

    res = pd.DataFrame(results)
    print(f"유효 분석일: {len(res)}일")

    # Triple confirmation at 15:00: Both QQQ, NVDA, SOXL fell between 14:30 and 15:00
    all_down = res[(res['qqq_pwr_in'] < -0.2) & (res['nvda_pwr_in'] < -0.3) & (res['soxl_pwr_in'] < -0.5)]
    print(f"\n[크로스에셋 하락 일치: 14:30~15:00 QQQ < -0.2% & NVDA < -0.3% & SOXL < -0.5% (N={len(all_down)}일)]")
    # SOXS profit
    soxs_win = (all_down['soxl_out'] < 0).mean() * 100
    print(f"  - 15:00~15:50 SOXS(추가 하락) 승률: {soxs_win:.1f}%")
    print(f"  - SOXS 평균 수익률: {-all_down['soxl_out'].mean():+.2f}%")

    # Triple confirmation at 15:00: Both QQQ, NVDA, SOXL rose between 14:30 and 15:00
    all_up = res[(res['qqq_pwr_in'] > 0.2) & (res['nvda_pwr_in'] > 0.3) & (res['soxl_pwr_in'] > 0.5)]
    print(f"\n[크로스에셋 상승 일치: 14:30~15:00 QQQ > 0.2% & NVDA > 0.3% & SOXL > 0.5% (N={len(all_up)}일)]")
    soxl_win = (all_up['soxl_out'] > 0).mean() * 100
    print(f"  - 15:00~15:50 SOXL(추가 상승) 승률: {soxl_win:.1f}%")
    print(f"  - SOXL 평균 수익률: {all_up['soxl_out'].mean():+.2f}%")

if __name__ == '__main__':
    test_cross_asset_at_close()
