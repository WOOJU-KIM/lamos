import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
from core.heterogeneous_models import CrossAssetDislocationModel

lake = MarketDataLake()
soxl_15m = lake.load_candles("SOXL", "15m")
soxs_15m = lake.load_candles("SOXS", "15m")
soxl_5m  = lake.load_candles("SOXL", "5m")
soxs_5m  = lake.load_candles("SOXS", "5m")
soxx_60m = lake.load_candles("SOXX", "60m")
soxl_60m = lake.load_candles("SOXL", "60m")
nvda_15m = lake.load_candles("NVDA", "15m")
qqq_15m  = lake.load_candles("QQQ", "15m")
vix_15m  = lake.load_candles("^VIX", "15m")

for df in [soxl_15m, soxs_15m, soxl_5m, soxs_5m, soxx_60m, soxl_60m, nvda_15m, qqq_15m, vix_15m]:
    df['datetime_dt'] = pd.to_datetime(df['datetime'])
    df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
    df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

unique_dates = sorted(soxl_15m['date_str'].unique())
soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
soxl_60m['ema20'] = soxl_60m['Close'].ewm(span=20, adjust=False).mean()

moe = MoEMetaOrchestrator(confidence_threshold=0.55, gbdt_threshold=0.55, mode="hybrid_v3")
soxl_15m_feat = moe.gbdt_engine.extract_features(soxl_15m)
soxl_15m_feat = moe.gbdt_engine.add_confidence_columns(soxl_15m_feat)
soxs_15m_feat = moe.gbdt_engine.extract_features(soxs_15m)
soxs_15m_feat.set_index('datetime', inplace=True, drop=False)

cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
cross_dirs, cross_confs = [], []
nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
vix_map  = vix_15m.set_index('datetime')['Close'].to_dict()
soxl_close_list = soxl_15m['Close'].values
soxl_dt_list = soxl_15m['datetime'].values

for i in range(len(soxl_15m)):
    if i < 5:
        cross_dirs.append("NONE")
        cross_confs.append(0.50)
        continue
    cur_t = soxl_dt_list[i]
    past_5_t = soxl_dt_list[i-5]
    s_r = float(soxl_close_list[i] / soxl_close_list[i-5] - 1.0)
    if (cur_t in nvda_map and past_5_t in nvda_map and 
        cur_t in qqq_map and past_5_t in qqq_map and 
        cur_t in vix_map and past_5_t in vix_map):
        n_r = float(nvda_map[cur_t] / nvda_map[past_5_t] - 1.0)
        q_r = float(qqq_map[cur_t] / qqq_map[past_5_t] - 1.0)
        v_r = float(vix_map[cur_t] / vix_map[past_5_t] - 1.0)
        sig_code, exp_conf, _ = cross_mod.predict_signal(
            soxl_ret=s_r, nvda_ret=n_r, qqq_ret=q_r, soxx_ret=s_r, vix_ret=v_r, tnx_ret=0.0
        )
        c_dir = "LONG_SOXL" if sig_code > 0 else ("SHORT_SOXS" if sig_code < 0 else "NONE")
        c_conf = exp_conf
    else:
        c_dir = "NONE"
        c_conf = 0.50
    cross_dirs.append(c_dir)
    cross_confs.append(c_conf)

soxl_15m_feat['cross_dir'] = cross_dirs
soxl_15m_feat['cross_conf'] = cross_confs
soxl_15m_feat.set_index('datetime', inplace=True, drop=False)

T = 0.55
TP_PCT = 0.035
SL_PCT = -0.020
TIME_STOP_BARS_5M = 18
SLIPPAGE_PAYUP = 0.03
FEE_RATE = 0.0020

trades = []
for d_str in unique_dates:
    day_soxl_15 = soxl_15m_feat[soxl_15m_feat['date_str'] == d_str]
    if len(day_soxl_15) < 5:
        continue
    day_soxl_5 = soxl_5m[soxl_5m['date_str'] == d_str]
    day_soxs_5 = soxs_5m[soxs_5m['date_str'] == d_str]
    daily_stoploss_count = 0
    b_idx = 0
    n_bars = len(day_soxl_15)

    while b_idx < n_bars:
        cur_15m_row = day_soxl_15.iloc[b_idx]
        cur_15m_time = cur_15m_row['datetime']
        time_str = cur_15m_row['time_str']

        if b_idx < 1 or time_str > "14:30" or daily_stoploss_count >= 3:
            b_idx += 1
            continue

        dir_gbdt = cur_15m_row['Direction']
        conf_gbdt = float(cur_15m_row['Confidence'])
        dir_cross = cur_15m_row['cross_dir']

        is_gbdt_trigger = (dir_gbdt in ["LONG_SOXL", "SHORT_SOXS"]) and (conf_gbdt >= T)
        is_cross_veto = (
            (dir_gbdt == "LONG_SOXL" and dir_cross == "SHORT_SOXS") or
            (dir_gbdt == "SHORT_SOXS" and dir_cross == "LONG_SOXL")
        )

        if not is_gbdt_trigger or is_cross_veto:
            b_idx += 1
            continue

        direction = dir_gbdt
        past_soxx_60 = soxx_60m[soxx_60m['datetime'] <= cur_15m_time]
        past_soxl_60 = soxl_60m[soxl_60m['datetime'] <= cur_15m_time]
        is_60m_trend_ok = True
        if len(past_soxx_60) >= 20 and len(past_soxl_60) >= 20:
            soxx_c = past_soxx_60['Close'].iloc[-1]
            soxl_c = past_soxl_60['Close'].iloc[-1]
            soxx_ema20 = past_soxx_60['ema20'].iloc[-1]
            soxl_ema20 = past_soxl_60['ema20'].iloc[-1]
            if direction == "LONG_SOXL":
                is_60m_trend_ok = (soxx_c >= soxx_ema20 * 0.998) and (soxl_c >= soxl_ema20 * 0.998)
            elif direction == "SHORT_SOXS":
                is_60m_trend_ok = (soxx_c <= soxx_ema20 * 1.002)

        if not is_60m_trend_ok:
            b_idx += 1
            continue

        if direction == "LONG_SOXL":
            vwap_diff = float(cur_15m_row.get("VWAP_Diff", 0.0))
            rsi_14 = float(cur_15m_row.get("RSI_14", 50.0))
            bb_lower = float(cur_15m_row.get("BB_Lower", 0.0))
            cur_close = float(cur_15m_row['Close'])
            dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
            if bb_lower > 0:
                dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
        else:
            if cur_15m_time in soxs_15m_feat.index:
                row_s = soxs_15m_feat.loc[cur_15m_time]
                vwap_diff = float(row_s.get("VWAP_Diff", 0.0))
                rsi_14 = float(row_s.get("RSI_14", 50.0))
                bb_lower = float(row_s.get("BB_Lower", 0.0))
                cur_close = float(row_s['Close'])
                dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                if bb_lower > 0:
                    dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
            else:
                dip_ok = False

        if not dip_ok:
            b_idx += 1
            continue

        chosen_symbol = "SOXL" if direction == "LONG_SOXL" else "SOXS"
        base_px = float(cur_15m_row['Close']) if chosen_symbol == "SOXL" else float(soxs_15m_feat.loc[cur_15m_time]['Close'] if cur_15m_time in soxs_15m_feat.index else 40.0)
        entry_px = round(base_px + SLIPPAGE_PAYUP, 2)

        target_5m_df = day_soxl_5 if chosen_symbol == "SOXL" else day_soxs_5
        post_5m = target_5m_df[target_5m_df['datetime'] > cur_15m_time]
        if post_5m.empty:
            b_idx += 1
            continue

        tp_px = round(entry_px * (1 + TP_PCT), 2)
        sl_px = round(entry_px * (1 + SL_PCT), 2)
        eval_5m = post_5m.iloc[:TIME_STOP_BARS_5M]

        # Detailed tracking of candle path
        had_intra_5m_collision = False
        min_drawdown_pct = 0.0
        max_runup_pct = 0.0
        exit_triggered = False
        exit_px = 0.0
        exit_reason = ""
        bars_held_5m = 0

        for k in range(len(eval_5m)):
            c5 = eval_5m.iloc[k]
            t5_str = c5['time_str']
            c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
            bars_held_5m = k + 1

            c_dd = (c5_l - entry_px) / entry_px
            c_ru = (c5_h - entry_px) / entry_px
            min_drawdown_pct = min(min_drawdown_pct, c_dd)
            max_runup_pct = max(max_runup_pct, c_ru)

            hit_tp = (c5_h >= tp_px)
            hit_sl = (c5_l <= sl_px)

            if hit_tp and hit_sl:
                had_intra_5m_collision = True
                if c5_c >= c5_o:
                    exit_triggered, exit_px, exit_reason = True, round(tp_px - SLIPPAGE_PAYUP, 2), "TP_RESCUED"
                    break
                else:
                    exit_triggered, exit_px, exit_reason = True, round(sl_px - SLIPPAGE_PAYUP, 2), "SL_COLLISION"
                    daily_stoploss_count += 1
                    break
            elif hit_tp:
                exit_triggered, exit_px, exit_reason = True, round(tp_px - SLIPPAGE_PAYUP, 2), "TP"
                break
            elif hit_sl:
                exit_triggered, exit_px, exit_reason = True, round(sl_px - SLIPPAGE_PAYUP, 2), "SL"
                daily_stoploss_count += 1
                break
            if t5_str >= "15:45":
                exit_triggered, exit_px, exit_reason = True, round(c5_c - SLIPPAGE_PAYUP, 2), "EOD"
                break
            if bars_held_5m >= TIME_STOP_BARS_5M:
                exit_triggered, exit_px, exit_reason = True, round(c5_c - SLIPPAGE_PAYUP, 2), "TIMESTOP"
                break

        if not exit_triggered and len(eval_5m) > 0:
            exit_px = round(float(eval_5m.iloc[-1]['Close']) - SLIPPAGE_PAYUP, 2)
            exit_reason = "EOD"

        pnl_pct = (exit_px - entry_px) / entry_px - FEE_RATE
        trades.append({
            "date": d_str,
            "symbol": chosen_symbol,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "exit_reason": exit_reason,
            "is_win": pnl_pct > 0,
            "pnl_pct": pnl_pct * 100,
            "had_intra_5m_collision": had_intra_5m_collision,
            "min_drawdown_pct": min_drawdown_pct * 100,
            "max_runup_pct": max_runup_pct * 100
        })
        bars_to_advance = max(1, int(np.ceil(bars_held_5m / 3.0)))
        b_idx += bars_to_advance

df_t = pd.DataFrame(trades)
print("Total trades at 55%:", len(df_t))
print("Current 5m Win Rate:", len(df_t[df_t['is_win']]) / len(df_t) * 100)

# Intra-5m collisions that were counted as wins
intra_rescued = df_t[df_t['exit_reason'] == 'TP_RESCUED']
print(f"Intra-5m rescued wins: {len(intra_rescued)} trades")

# Near-stoploss wins (min_drawdown between -1.5% and -1.99%)
near_sl_wins = df_t[(df_t['is_win']) & (df_t['min_drawdown_pct'] <= -1.5)]
print(f"Near-stoploss wins (low dropped between -1.5% and -2.0%): {len(near_sl_wins)} trades")

# TimeStop trades that barely won
barely_won_timestop = df_t[(df_t['is_win']) & (df_t['exit_reason'] == 'TIMESTOP') & (df_t['pnl_pct'] < 1.0)]
print(f"Barely won timestop (< +1.0%): {len(barely_won_timestop)} trades")
