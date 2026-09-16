python -c "
import sys
from pathlib import Path
sys.path.append(r'C:\Users\chabo\.gemini\antigravity\brain\6bd29a15-5cae-4a5f-8c66-b631de55eb9b\scratch')
from backtest_hybrid_moe_v3_doe import load_lumos_dataset, generate_hybrid_signals
import pandas as pd
import numpy as np

df = load_lumos_dataset()
signals, veto_mask = generate_hybrid_signals(df, 0.55)

# Monthly breakdown of trades
TP_PCT = 0.035
SL_PCT = -0.020
TIME_STOP_BARS = 6
SLIPPAGE_PAYUP = 0.03
FEE_RATE = 0.0020
capital = 10_000_000.0
trades = []
active_pos = None

for d_str in sorted(df['date'].unique()):
    day_df = df[df['date'] == d_str]
    if len(day_df) < 5:
        continue
    for b_idx in range(len(day_df)):
        row = day_df.iloc[b_idx]
        time_str = row['time']
        dt = row.name

        if active_pos is not None:
            bars_held = b_idx - active_pos['entry_bar_idx'] + 1
            sym = active_pos['symbol']
            buy_px = active_pos['buy_price']
            cur_h = row['high'] if sym == 'SOXL' else row['soxs_high']
            cur_l = row['low'] if sym == 'SOXL' else row['soxs_low']
            cur_c = row['close'] if sym == 'SOXL' else row['soxs_close']

            max_ret = (cur_h - buy_px) / buy_px
            min_ret = (cur_l - buy_px) / buy_px

            exit_triggered = False
            exit_reason = ''
            exit_px = 0.0

            if max_ret >= TP_PCT:
                exit_triggered = True
                exit_reason = 'TP (+3.5%)'
                exit_px = round(buy_px * (1 + TP_PCT) - SLIPPAGE_PAYUP, 2)
            elif min_ret <= SL_PCT:
                exit_triggered = True
                exit_reason = 'SL (-2.0%)'
                exit_px = round(buy_px * (1 + SL_PCT) - SLIPPAGE_PAYUP, 2)
            elif bars_held >= TIME_STOP_BARS:
                exit_triggered = True
                exit_reason = 'TimeStop (90m)'
                exit_px = round(cur_c - SLIPPAGE_PAYUP, 2)
            elif b_idx >= len(day_df) - 1 or time_str >= '15:45':
                exit_triggered = True
                exit_reason = 'EOD Close'
                exit_px = round(cur_c - SLIPPAGE_PAYUP, 2)

            if exit_triggered:
                invested = active_pos['invested']
                shares = active_pos['shares']
                cost_amount = invested * FEE_RATE
                pnl_amount = (shares * (exit_px - buy_px)) - cost_amount
                capital += pnl_amount
                trades.append({
                    'month': dt.strftime('%Y-%m'),
                    'symbol': sym,
                    'is_win': pnl_amount > 0,
                    'pnl_amount': pnl_amount,
                    'end_capital': capital
                })
                active_pos = None
                continue

        if active_pos is None and time_str <= '14:30' and b_idx >= 1:
            sig = signals.loc[dt]
            if sig in ['SOXL', 'SOXS']:
                trend_ok = row['trend_ok_soxl'] if sig == 'SOXL' else row['trend_ok_soxs']
                if not trend_ok:
                    continue
                vwap_diff = row['VWAP_Diff']
                rsi_14 = row['RSI_14']
                bb_lower = row['BB_Lower']
                cur_c = row['close']
                dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                if bb_lower > 0:
                    dip_ok = dip_ok and (cur_c >= bb_lower * 1.001)
                if not dip_ok:
                    continue

                base_px = row['close'] if sig == 'SOXL' else row['soxs_close']
                entry_px = round(base_px + SLIPPAGE_PAYUP, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares > 0 and invested > 0:
                    active_pos = {
                        'symbol': sig,
                        'entry_bar_idx': b_idx,
                        'buy_price': entry_px,
                        'shares': shares,
                        'invested': invested
                    }

df_tr = pd.DataFrame(trades)
for m in sorted(df_tr['month'].unique()):
    m_df = df_tr[df_tr['month'] == m]
    wins = len(m_df[m_df['is_win']])
    losses = len(m_df[~m_df['is_win']])
    wr = wins / len(m_df) * 100
    pnl = m_df['pnl_amount'].sum()
    end_cap = m_df['end_capital'].iloc[-1]
    print(f'{m}: {len(m_df)}회 | {wins}승 {losses}패 (승률 {wr:.1f}%) | 순손익 {pnl:+,.0f}원 | 월말잔액 {end_cap:,.0f}원')
"