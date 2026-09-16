import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lumos")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine

def run_comparison():
    lake = MarketDataLake()
    
    # 1. Load data
    soxl_15m = lake.load_candles("SOXL", "15m")
    soxs_15m = lake.load_candles("SOXS", "15m")
    soxl_5m  = lake.load_candles("SOXL", "5m")
    soxs_5m  = lake.load_candles("SOXS", "5m")
    soxx_60m = lake.load_candles("SOXX", "60m")
    soxl_60m = lake.load_candles("SOXL", "60m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vix_15m  = lake.load_candles("^VIX", "15m")

    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
    soxl_60m['ema20'] = soxl_60m['Close'].ewm(span=20, adjust=False).mean()

    for df in [soxl_15m, soxs_15m, soxl_5m, soxs_5m, soxx_60m, soxl_60m, soxx_15m, nvda_15m, qqq_15m, vix_15m]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    # Features
    ml_15m = MLFeatureEngine()
    soxl_15m_feat = ml_15m.extract_features(soxl_15m)
    soxl_15m_feat = ml_15m.add_confidence_columns(soxl_15m_feat)
    soxs_15m_feat = ml_15m.extract_features(soxs_15m)
    soxs_15m_feat.set_index('datetime', inplace=True, drop=False)

    # Cross asset model
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map = vix_15m.set_index('datetime')['Close'].to_dict()
    soxl_close = soxl_15m['Close'].values
    soxl_dt = soxl_15m['datetime'].values

    cross_dirs = []
    cross_confs = []
    for i in range(len(soxl_15m)):
        if i < 5:
            cross_dirs.append('NONE'); cross_confs.append(0.50); continue
        c_t, p_t = soxl_dt[i], soxl_dt[i-5]
        if c_t in nvda_map and p_t in nvda_map and c_t in qqq_map and p_t in qqq_map and c_t in vix_map and p_t in vix_map:
            s_r = float(soxl_close[i]/soxl_close[i-5] - 1.0)
            n_r = float(nvda_map[c_t]/nvda_map[p_t] - 1.0)
            sx_r = float(soxx_map[c_t]/soxx_map[p_t] - 1.0) if c_t in soxx_map and p_t in soxx_map else n_r
            q_r = float(qqq_map[c_t]/qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t]/vix_map[p_t] - 1.0)
            code, conf, desc = cross_mod.predict_signal(soxl_ret=s_r, nvda_ret=n_r, soxx_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            c_dir = 'LONG_SOXL' if code > 0 else ('SHORT_SOXS' if code < 0 else 'NONE')
            cross_dirs.append(c_dir)
            cross_confs.append(conf)
        else:
            cross_dirs.append('NONE'); cross_confs.append(0.50)

    soxl_15m_feat['cross_dir'] = cross_dirs
    soxl_15m_feat['cross_conf'] = cross_confs
    soxl_15m_feat.set_index('datetime', inplace=True, drop=False)

    unique_dates = sorted(soxl_15m['date_str'].unique())

    TP_15M = 0.030
    SL_15M = -0.020
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    TIME_STOP_BARS = 18  # 90 minutes (18 5m bars)

    def simulate(mode: str, use_screen1: bool, use_screen3: bool, use_tp_sl: bool = True):
        capital = 10000.0
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        for d_str in unique_dates:
            day_15 = soxl_15m_feat[soxl_15m_feat['date_str'] == d_str]
            day_5_l = soxl_5m[soxl_5m['date_str'] == d_str]
            day_5_s = soxs_5m[soxs_5m['date_str'] == d_str]
            if len(day_15) < 5:
                continue

            daily_stoploss_count = 0
            b_idx = 0
            n_bars = len(day_15)

            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                if b_idx < 1 or time_str > '14:30' or daily_stoploss_count >= 3:
                    b_idx += 1
                    continue

                dir_gbdt = row_15['Direction']
                conf_gbdt = float(row_15['Confidence'])
                dir_cross = row_15['cross_dir']
                conf_cross = float(row_15['cross_conf'])

                chosen_sym = None
                if mode == "gbdt_only":
                    # 순수 GBDT 점수 >= 60%만 사용 (배리어/필터 없음)
                    if dir_gbdt in ['LONG_SOXL', 'SHORT_SOXS'] and conf_gbdt >= 0.60:
                        chosen_sym = 'SOXL' if dir_gbdt == 'LONG_SOXL' else 'SOXS'

                elif mode == "gbdt_cross_veto":
                    # GBDT 60% + 크로스에셋 반대방향 VETO (정반대 거시 역풍 차단)
                    is_gbdt = (dir_gbdt in ['LONG_SOXL', 'SHORT_SOXS']) and (conf_gbdt >= 0.60)
                    is_opposite_veto = (
                        (dir_gbdt == 'LONG_SOXL' and dir_cross == 'SHORT_SOXS' and conf_cross >= 0.60) or
                        (dir_gbdt == 'SHORT_SOXS' and dir_cross == 'LONG_SOXL' and conf_cross >= 0.60)
                    )
                    if is_gbdt and not is_opposite_veto:
                        chosen_sym = 'SOXL' if dir_gbdt == 'LONG_SOXL' else 'SOXS'

                elif mode == "hybrid_v3":
                    # 현재 챔피언: GBDT 60% + Cross-Asset Veto + Screen 1 + Screen 3
                    is_gbdt = (dir_gbdt in ['LONG_SOXL', 'SHORT_SOXS']) and (conf_gbdt >= 0.60)
                    is_opposite_veto = (
                        (dir_gbdt == 'LONG_SOXL' and dir_cross == 'SHORT_SOXS' and conf_cross >= 0.60) or
                        (dir_gbdt == 'SHORT_SOXS' and dir_cross == 'LONG_SOXL' and conf_cross >= 0.60)
                    )
                    if is_gbdt and not is_opposite_veto:
                        chosen_sym = 'SOXL' if dir_gbdt == 'LONG_SOXL' else 'SOXS'

                elif mode == "consensus_legacy":
                    # 현재 레거시 안전 모델: GBDT 60% AND Cross-Asset 60% 100% 동시 일치
                    if dir_gbdt == 'LONG_SOXL' and dir_cross == 'LONG_SOXL' and conf_gbdt >= 0.60 and conf_cross >= 0.60:
                        chosen_sym = 'SOXL'
                    elif dir_gbdt == 'SHORT_SOXS' and dir_cross == 'SHORT_SOXS' and conf_gbdt >= 0.60 and conf_cross >= 0.60:
                        chosen_sym = 'SOXS'

                if not chosen_sym:
                    b_idx += 1
                    continue

                # Screen 1 (60분봉 대추세 필터)
                if use_screen1:
                    past_soxx = soxx_60m[soxx_60m['datetime'] <= cur_time]
                    past_soxl = soxl_60m[soxl_60m['datetime'] <= cur_time]
                    if len(past_soxx) < 20 or len(past_soxl) < 20:
                        b_idx += 1; continue
                    soxx_c = past_soxx['Close'].iloc[-1]
                    soxl_c = past_soxl['Close'].iloc[-1]
                    soxx_ema = past_soxx['ema20'].iloc[-1]
                    soxl_ema = past_soxl['ema20'].iloc[-1]
                    if chosen_sym == 'SOXL' and not (soxx_c >= soxx_ema * 0.998 and soxl_c >= soxl_ema * 0.998):
                        b_idx += 1; continue
                    elif chosen_sym == 'SOXS' and not (soxx_c <= soxx_ema * 1.002):
                        b_idx += 1; continue

                # Screen 3 (단기 눌림목/과열 필터)
                if use_screen3:
                    if chosen_sym == 'SOXL':
                        vd = float(row_15.get('VWAP_Diff', 0.0))
                        r14 = float(row_15.get('RSI_14', 50.0))
                        bbl = float(row_15.get('BB_Lower', 0.0))
                        if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(row_15['Close']) >= bbl * 1.001)):
                            b_idx += 1; continue
                    else:
                        if cur_time not in soxs_15m_feat.index:
                            b_idx += 1; continue
                        rs = soxs_15m_feat.loc[cur_time]
                        vd = float(rs.get('VWAP_Diff', 0.0))
                        r14 = float(rs.get('RSI_14', 50.0))
                        bbl = float(rs.get('BB_Lower', 0.0))
                        if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(rs['Close']) >= bbl * 1.001)):
                            b_idx += 1; continue

                if chosen_sym == 'SOXS' and cur_time not in soxs_15m_feat.index:
                    b_idx += 1
                    continue

                base_px = float(row_15['Close']) if chosen_sym == 'SOXL' else float(soxs_15m_feat.loc[cur_time]['Close'])
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0:
                    b_idx += 1; continue

                target_5m = day_5_l if chosen_sym == 'SOXL' else day_5_s
                post_5m = target_5m[target_5m['datetime'] > cur_time]
                if post_5m.empty:
                    b_idx += 1; continue

                tp_px = round(entry_px * (1 + TP_15M), 2)
                sl_px = round(entry_px * (1 + SL_15M), 2)
                exit_px = entry_px
                exit_reason = "TIMESTOP"
                bars_held = 0

                if use_tp_sl:
                    eval_5m = post_5m.iloc[:TIME_STOP_BARS]
                    for k in range(len(eval_5m)):
                        c5 = eval_5m.iloc[k]
                        t5 = c5['time_str']
                        c5_h, c5_l, c5_c = float(c5['High']), float(c5['Low']), float(c5['Close'])
                        bars_held = k + 1
                        hit_tp, hit_sl = (c5_h >= tp_px), (c5_l <= sl_px)

                        if hit_tp and not hit_sl:
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            exit_reason = "TP (+3.0%)"
                            break
                        elif hit_sl and not hit_tp:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = "SL (-2.0%)"
                            daily_stoploss_count += 1
                            break
                        elif hit_tp and hit_sl:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = "SL (-2.0% 변동성)"
                            daily_stoploss_count += 1
                            break
                        elif t5 >= '15:50':
                            exit_px = round(c5_c - SLIPPAGE, 2)
                            exit_reason = "EOD (15:50 0% 현금화)"
                            break
                    else:
                        exit_px = round(float(eval_5m.iloc[-1]['Close']) - SLIPPAGE, 2)
                        exit_reason = "TIMESTOP (90분 만기)"
                else:
                    # No TP/SL barriers: exit only on opposite 15m signal or EOD 15:50
                    post_15 = day_15.iloc[b_idx+1:]
                    for k15 in range(len(post_15)):
                        r15 = post_15.iloc[k15]
                        t15 = r15['time_str']
                        d15 = r15['Direction']
                        # if signal reverses
                        opp_sig = 'SHORT_SOXS' if chosen_sym == 'SOXL' else 'LONG_SOXL'
                        if d15 == opp_sig or t15 >= '15:50':
                            if chosen_sym == 'SOXL':
                                c15_close = float(r15['Close'])
                            else:
                                if r15['datetime'] in soxs_15m_feat.index:
                                    c15_close = float(soxs_15m_feat.loc[r15['datetime']]['Close'])
                                else:
                                    c15_close = float(soxs_15m_feat.iloc[-1]['Close'])
                            exit_px = round(c15_close - SLIPPAGE, 2)
                            exit_reason = "신호 반전 청산" if d15 == opp_sig else "EOD (15:50)"
                            bars_held = (k15 + 1) * 3
                            break
                    else:
                        c_last = float(day_15.iloc[-1]['Close']) if chosen_sym == 'SOXL' else float(soxs_15m_feat.iloc[-1]['Close'])
                        exit_px = round(c_last - SLIPPAGE, 2)
                        exit_reason = "EOD (장마감)"
                        bars_held = len(post_15) * 3

                # PnL calc
                gross_ret = (exit_px - entry_px) / entry_px
                net_ret = gross_ret - FEE_RATE
                pnl_dollars = invested * net_ret
                capital += pnl_dollars
                equity_curve.append(capital)

                trades.append({
                    'date': d_str,
                    'time': time_str,
                    'symbol': chosen_sym,
                    'entry_px': entry_px,
                    'exit_px': exit_px,
                    'ret': net_ret,
                    'pnl': pnl_dollars,
                    'exit_reason': exit_reason,
                    'is_win': net_ret > 0,
                    'bars_held': bars_held
                })

                # skip forward in 15m bars
                skip_15m_bars = max(1, int(np.ceil(bars_held / 3.0)))
                b_idx += skip_15m_bars

        # Summary stats
        df_tr = pd.DataFrame(trades)
        total_trades = len(df_tr)
        if total_trades == 0:
            return {
                "total_trades": 0, "win_rate": 0.0, "pnl_dollars": 0.0,
                "profit_factor": 0.0, "max_drawdown": 0.0, "cagr": 0.0
            }

        wins = df_tr[df_tr['ret'] > 0]
        losses = df_tr[df_tr['ret'] <= 0]
        win_rate = len(wins) / total_trades * 100.0

        total_gain = wins['pnl'].sum() if not wins.empty else 0.0
        total_loss = abs(losses['pnl'].sum()) if not losses.empty else 0.0
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (99.9 if total_gain > 0 else 0.0)

        # Max Drawdown
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak * 100.0
        mdd = np.max(dd)

        total_return_pct = (capital - 10000.0) / 10000.0 * 100.0

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "final_capital": round(capital, 2),
            "total_return_pct": round(total_return_pct, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown": round(mdd, 2),
            "avg_trade_ret": round(df_tr['ret'].mean() * 100.0, 2),
            "wins_count": len(wins),
            "losses_count": len(losses)
        }

    print("Running Backtests...")
    # 1. Pure GBDT only (No Screen 1, No Screen 3, No Veto) - With TP/SL
    res1 = simulate(mode="gbdt_only", use_screen1=False, use_screen3=False, use_tp_sl=True)
    print("1. GBDT Only (No filter barrier):", res1)

    # 2. GBDT + Cross-Asset Veto (No Screen 1, No Screen 3) - With TP/SL
    res2 = simulate(mode="gbdt_cross_veto", use_screen1=False, use_screen3=False, use_tp_sl=True)
    print("2. GBDT + Cross-Asset Veto (No screen barrier):", res2)

    # 3. Current Champion Model: Lumos V3 Hybrid MoE (All barriers: Screen 1 + Screen 3 + Veto) - With TP/SL
    res3 = simulate(mode="hybrid_v3", use_screen1=True, use_screen3=True, use_tp_sl=True)
    print("3. Current Champion (Lumos V3 Hybrid MoE with all barriers):", res3)

    # 4. Current Legacy Safe Model: Dual Consensus AND (All barriers) - With TP/SL
    res4 = simulate(mode="consensus_legacy", use_screen1=True, use_screen3=True, use_tp_sl=True)
    print("4. Current Legacy Safe (Dual Consensus AND with all barriers):", res4)

    # 5. Pure GBDT only WITHOUT TP/SL barrier (Signal reverse exit)
    res5 = simulate(mode="gbdt_only", use_screen1=False, use_screen3=False, use_tp_sl=False)
    print("5. Pure GBDT WITHOUT TP/SL Barrier (Reverse exit):", res5)

    print("\n--- Summary Results ---")
    results = {
        "1_gbdt_only_no_filter": res1,
        "2_gbdt_cross_veto_only": res2,
        "3_current_hybrid_v3": res3,
        "4_current_legacy_consensus": res4,
        "5_gbdt_no_tpsl_barrier": res5
    }
    import json
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_comparison()
