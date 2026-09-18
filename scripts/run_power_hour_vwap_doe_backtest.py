import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Set utf-8 stdout for Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def run_power_hour_vwap_sniper_doe():
    print("=" * 115)
    print("🏛️ [Lumos V3: Power Hour VWAP Sniper 4-Stage DOE Shadow Backtest]")
    print(f"⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 115)
    
    lake = MarketDataLake()
    print("⏳ [1/4] Loading multi-timeframe candle data from MarketDataLake...")
    
    tqqq_15m = lake.load_candles("TQQQ", "15m")
    sqqq_15m = lake.load_candles("SQQQ", "15m")
    tqqq_5m  = lake.load_candles("TQQQ", "5m")
    sqqq_5m  = lake.load_candles("SQQQ", "5m")
    soxx_60m = lake.load_candles("SOXX", "60m")
    tqqq_60m = lake.load_candles("TQQQ", "60m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vix_15m  = lake.load_candles("^VIX", "15m")

    # Precalculate EMA20 on 60m for Screen 1
    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
    tqqq_60m['ema20'] = tqqq_60m['Close'].ewm(span=20, adjust=False).mean()

    for df in [tqqq_15m, sqqq_15m, tqqq_5m, sqqq_5m, soxx_60m, tqqq_60m, soxx_15m, nvda_15m, qqq_15m, vix_15m]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    print("⏳ [2/4] Engineering ML Features and Cross-Asset Causal Vectors...")
    ml_15m = MLFeatureEngine()
    tqqq_15m_feat = ml_15m.extract_features(tqqq_15m)
    tqqq_15m_feat = ml_15m.add_confidence_columns(tqqq_15m_feat)
    sqqq_15m_feat = ml_15m.extract_features(sqqq_15m)
    sqqq_15m_feat.set_index('datetime', inplace=True, drop=False)

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map = vix_15m.set_index('datetime')['Close'].to_dict()
    tqqq_close = tqqq_15m['Close'].values
    tqqq_dt = tqqq_15m['datetime'].values

    cross_dirs = []
    cross_confs = []
    for i in range(len(tqqq_15m)):
        if i < 5:
            cross_dirs.append('NONE'); cross_confs.append(0.50); continue
        c_t, p_t = tqqq_dt[i], tqqq_dt[i-5]
        if c_t in nvda_map and p_t in nvda_map and c_t in qqq_map and p_t in qqq_map and c_t in vix_map and p_t in vix_map:
            s_r = float(tqqq_close[i]/tqqq_close[i-5] - 1.0)
            n_r = float(nvda_map[c_t]/nvda_map[p_t] - 1.0)
            sx_r = float(soxx_map[c_t]/soxx_map[p_t] - 1.0) if c_t in soxx_map and p_t in soxx_map else n_r
            q_r = float(qqq_map[c_t]/qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t]/vix_map[p_t] - 1.0)
            code, conf, _ = cross_mod.predict_signal(tqqq_ret=s_r, nvda_ret=n_r, soxx_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            c_dir = 'LONG_TQQQ' if code > 0 else ('SHORT_SQQQ' if code < 0 else 'NONE')
            cross_dirs.append(c_dir); cross_confs.append(conf)
        else:
            cross_dirs.append('NONE'); cross_confs.append(0.50)

    tqqq_15m_feat['cross_dir'] = cross_dirs
    tqqq_15m_feat['cross_conf'] = cross_confs
    tqqq_15m_feat.set_index('datetime', inplace=True, drop=False)

    # Calculate Intraday Cumulative VWAP for TQQQ on 5m and 15m
    # VWAP resets at 09:30 every single trading day
    print("⏳ [3/4] Pre-calculating Intraday Cumulative Session VWAP (09:30 Reset)...")
    tqqq_5m['cum_vol'] = tqqq_5m.groupby('date_str')['Volume'].cumsum()
    tqqq_5m['cum_pv'] = tqqq_5m.groupby('date_str').apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    tqqq_5m['intraday_vwap'] = tqqq_5m['cum_pv'] / (tqqq_5m['cum_vol'] + 1e-6)

    # Map 5m latest VWAP to 15m datetime
    vwap_5m_map = tqqq_5m.set_index('datetime')['intraday_vwap'].to_dict()

    unique_dates = sorted(tqqq_15m['date_str'].unique())

    # Fixed Execution Constants
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    TIME_STOP_BARS = 18  # 90 minutes (18 * 5m)

    # Phase 1 Parameters (Fixed Baseline)
    P1_TP = 0.030
    P1_SL = -0.020

    # Phase 2 DOE Parameters (Risk-Reward 1.5 : 1)
    doe_configs = [
        {"case_id": "Baseline", "name": "기존 Model C (14:30 셧다운)", "enable_p2": False, "p2_tp": 0.0, "p2_sl": 0.0},
        {"case_id": "Case 1", "name": "초단타 숏게임 (TP 1.5% / SL 1.0%)", "enable_p2": True, "p2_tp": 0.015, "p2_sl": -0.010},
        {"case_id": "Case 2", "name": "스탠다드 (TP 2.0% / SL 1.33%)", "enable_p2": True, "p2_tp": 0.020, "p2_sl": -0.0133},
        {"case_id": "Case 3", "name": "미들급 (TP 2.5% / SL 1.66%)", "enable_p2": True, "p2_tp": 0.025, "p2_sl": -0.0166},
        {"case_id": "Case 4", "name": "기존 밴드 유지 (TP 3.0% / SL 2.0%)", "enable_p2": True, "p2_tp": 0.030, "p2_sl": -0.020},
    ]

    def simulate_doe_case(cfg: Dict[str, Any]) -> Dict[str, Any]:
        capital = 10000.0
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        enable_p2 = cfg["enable_p2"]
        p2_tp = cfg["p2_tp"]
        p2_sl = cfg["p2_sl"]

        for d_str in unique_dates:
            day_15 = tqqq_15m_feat[tqqq_15m_feat['date_str'] == d_str]
            day_5_l = tqqq_5m[tqqq_5m['date_str'] == d_str]
            day_5_s = sqqq_5m[sqqq_5m['date_str'] == d_str]
            if len(day_15) < 5:
                continue

            b_idx = 0
            n_bars = len(day_15)
            daily_stoploss_count = 0

            # Track intra-day position state for single-position relay
            # If position opened before 14:30 and still held past 14:30, Phase 2 cannot enter
            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                # Circuit breaker check
                if daily_stoploss_count >= 3:
                    b_idx += 1; continue

                # Session classification
                is_p1 = ('09:30' <= time_str <= '14:30')
                is_p2 = ('14:30' < time_str <= '15:20')

                if not is_p1 and not (enable_p2 and is_p2):
                    b_idx += 1; continue

                phase_tag = "Phase 1" if is_p1 else "Phase 2 (VWAP Sniper)"

                # Core GBDT and Veto Interlocks
                dir_gbdt = row_15['Direction']
                conf_gbdt = float(row_15['Confidence'])
                dir_cross = row_15['cross_dir']
                conf_cross = float(row_15['cross_conf'])

                is_gbdt = (dir_gbdt in ['LONG_TQQQ', 'SHORT_SQQQ']) and (conf_gbdt >= 0.60)
                is_opposite_veto = (
                    (dir_gbdt == 'LONG_TQQQ' and dir_cross == 'SHORT_SQQQ' and conf_cross >= 0.60) or
                    (dir_gbdt == 'SHORT_SQQQ' and dir_cross == 'LONG_TQQQ' and conf_cross >= 0.60)
                )

                if not (is_gbdt and not is_opposite_veto):
                    b_idx += 1; continue

                chosen_sym = 'TQQQ' if dir_gbdt == 'LONG_TQQQ' else 'SQQQ'

                # Phase 2 VWAP Directional Lock Filter
                if is_p2:
                    current_vwap = vwap_5m_map.get(cur_time, None)
                    if current_vwap is None or current_vwap <= 0:
                        # Fallback to 15m close if 5m vwap not mapped
                        past_5 = day_5_l[day_5_l['datetime'] <= cur_time]
                        if not past_5.empty and past_5['cum_vol'].iloc[-1] > 0:
                            current_vwap = past_5['cum_pv'].iloc[-1] / past_5['cum_vol'].iloc[-1]
                        else:
                            current_vwap = float(row_15['Close'])

                    tqqq_spot = float(row_15['Close'])
                    # VWAP Lock: Spot > VWAP -> ONLY TQQQ allowed; Spot < VWAP -> ONLY SQQQ allowed
                    if tqqq_spot >= current_vwap:
                        if chosen_sym != 'TQQQ':
                            # Reject SQQQ signal when price is above VWAP
                            b_idx += 1; continue
                    else:
                        if chosen_sym != 'SQQQ':
                            # Reject TQQQ signal when price is below VWAP
                            b_idx += 1; continue

                # Screen 1: 60m trend
                past_soxx = soxx_60m[soxx_60m['datetime'] <= cur_time]
                past_tqqq = tqqq_60m[tqqq_60m['datetime'] <= cur_time]
                if len(past_soxx) < 20 or len(past_tqqq) < 20:
                    b_idx += 1; continue
                soxx_c = past_soxx['Close'].iloc[-1]
                tqqq_c = past_tqqq['Close'].iloc[-1]
                soxx_ema = past_soxx['ema20'].iloc[-1]
                tqqq_ema = past_tqqq['ema20'].iloc[-1]
                if chosen_sym == 'TQQQ' and not (soxx_c >= soxx_ema * 0.998 and tqqq_c >= tqqq_ema * 0.998):
                    b_idx += 1; continue
                elif chosen_sym == 'SQQQ' and not (soxx_c <= soxx_ema * 1.002):
                    b_idx += 1; continue

                # Screen 3: Dip filter
                if chosen_sym == 'TQQQ':
                    vd = float(row_15.get('VWAP_Diff', 0.0))
                    r14 = float(row_15.get('RSI_14', 50.0))
                    bbl = float(row_15.get('BB_Lower', 0.0))
                    if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(row_15['Close']) >= bbl * 1.001)):
                        b_idx += 1; continue
                else:
                    if cur_time not in sqqq_15m_feat.index:
                        b_idx += 1; continue
                    rs = sqqq_15m_feat.loc[cur_time]
                    vd = float(rs.get('VWAP_Diff', 0.0))
                    r14 = float(rs.get('RSI_14', 50.0))
                    bbl = float(rs.get('BB_Lower', 0.0))
                    if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(rs['Close']) >= bbl * 1.001)):
                        b_idx += 1; continue

                # Entry execution with slippage
                base_px = float(row_15['Close']) if chosen_sym == 'TQQQ' else float(sqqq_15m_feat.loc[cur_time]['Close'])
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0:
                    b_idx += 1; continue

                target_5m = day_5_l if chosen_sym == 'TQQQ' else day_5_s
                post_5m = target_5m[target_5m['datetime'] > cur_time]
                if post_5m.empty:
                    b_idx += 1; continue

                # TP/SL Targets based on Phase
                cur_tp_pct = P1_TP if is_p1 else p2_tp
                cur_sl_pct = P1_SL if is_p1 else p2_sl

                tp_px = round(entry_px * (1 + cur_tp_pct), 2)
                sl_px = round(entry_px * (1 + cur_sl_pct), 2)
                exit_px = entry_px
                exit_reason = "TimeStop"
                exit_time_str = ""
                bars_held = 0

                eval_5m = post_5m.iloc[:TIME_STOP_BARS]
                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    t5 = c5['time_str']
                    c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
                    bars_held = k + 1
                    hit_tp, hit_sl = (c5_h >= tp_px), (c5_l <= sl_px)

                    # Intra-bar priority resolution (dual touch)
                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            exit_reason = f"TP +{cur_tp_pct*100:.1f}%"
                        else:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = f"SL {cur_sl_pct*100:.1f}%"
                            daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = f"TP +{cur_tp_pct*100:.1f}%"
                        exit_time_str = t5
                        break
                    elif hit_sl:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = f"SL {cur_sl_pct*100:.1f}%"
                        daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif t5 >= '15:45' or k == len(eval_5m) - 1:
                        # 15:50 EOD MOC Forced Sprint Liquidation
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        exit_reason = "15:50 EOD" if t5 >= '15:45' else "TimeStop 90m"
                        exit_time_str = t5
                        if (exit_px - entry_px) / entry_px <= cur_sl_pct:
                            daily_stoploss_count += 1
                        break

                cost = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost
                capital += net_pnl
                equity_curve.append(capital)
                ret = net_pnl / invested

                trades.append({
                    "date": d_str,
                    "symbol": chosen_sym,
                    "phase": phase_tag,
                    "entry_time": cur_time,
                    "entry_t": time_str,
                    "exit_t": exit_time_str,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "ret_pct": round(ret * 100, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "holding_minutes": bars_held * 5
                })

                # Single position relay jump: lock scanner until position liquidation
                consumed = int(np.ceil(bars_held / 3.0))
                b_idx += max(1, consumed)

        # Performance Aggregation
        tot_tr = len(trades)
        w_tr = [t for t in trades if t['net_pnl'] > 0]
        l_tr = [t for t in trades if t['net_pnl'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0

        p1_trades = [t for t in trades if t['phase'] == 'Phase 1']
        p2_trades = [t for t in trades if 'Phase 2' in t['phase']]

        p1_wins = [t for t in p1_trades if t['net_pnl'] > 0]
        p1_wr = (len(p1_wins) / len(p1_trades) * 100) if p1_trades else 0.0
        p1_pnl = sum([t['net_pnl'] for t in p1_trades])

        p2_wins = [t for t in p2_trades if t['net_pnl'] > 0]
        p2_wr = (len(p2_wins) / len(p2_trades) * 100) if p2_trades else 0.0
        p2_pnl = sum([t['net_pnl'] for t in p2_trades])

        gross_win = sum([t['ret_pct'] for t in w_tr])
        gross_loss = abs(sum([t['ret_pct'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9
        cum_ret = ((capital - 10000.0) / 10000.0) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        return {
            "case_id": cfg["case_id"],
            "name": cfg["name"],
            "total_trades": tot_tr,
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "profit_factor": pf,
            "cum_return": cum_ret,
            "mdd": mdd,
            "final_capital": capital,
            "p1_trades": len(p1_trades),
            "p1_win_rate": p1_wr,
            "p1_pnl": p1_pnl,
            "p2_trades": len(p2_trades),
            "p2_wins": len(p2_wins),
            "p2_losses": len(p2_trades) - len(p2_wins),
            "p2_win_rate": p2_wr,
            "p2_pnl": p2_pnl,
            "trades": trades
        }

    # Execute all 5 DOE cases (Baseline + Case 1..4)
    print("⏳ [4/4] Executing 4-Stage DOE Simulations (Parallel Session Accounting)...")
    results = []
    for cfg in doe_configs:
        res = simulate_doe_case(cfg)
        results.append(res)
        print(f"   ✓ Completed: [{res['case_id']}] {res['name']} ➔ Total Trades: {res['total_trades']} | P2 Trades: {res['p2_trades']} | Cum Return: {res['cum_return']:+.2f}%")

    base_res = results[0]
    base_cum_ret = base_res['cum_return']
    base_final_cap = base_res['final_capital']

    # Build Output Comparison DataFrame
    summary_rows = []
    for r in results:
        delta_pnl_pct = r['cum_return'] - base_cum_ret
        delta_usd = r['final_capital'] - base_final_cap
        summary_rows.append({
            "실험 케이스": r['case_id'],
            "파워아워 익/손절 밴드": r['name'],
            "전체 거래": f"{r['total_trades']}회",
            "전체 승률": f"{r['win_rate']:.1f}%",
            "전체 PF": f"{r['profit_factor']:.2f}",
            "누적 수익률": f"{r['cum_return']:+.2f}%",
            "MDD": f"{r['mdd']:.2f}%",
            "P2(추가) 거래수": f"{r['p2_trades']}회",
            "P2(추가) 승률": f"{r['p2_win_rate']:.1f}% ({r['p2_wins']}승/{r['p2_losses']}패)" if r['p2_trades'] > 0 else "-",
            "P2 추가 PnL ($)": f"{r['p2_pnl']:+,.2f}$" if r['p2_trades'] > 0 else "$0.00",
            "Delta 수익률(%)": f"{delta_pnl_pct:+.2f}%p" if r['case_id'] != 'Baseline' else "기준(Base)"
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 135)
    print("🏆 [Lumos V3: Power Hour VWAP Sniper 4-Stage DOE 백테스트 종합 결과표]")
    print("=" * 135)
    print(summary_df.to_string(index=False))
    print("=" * 135)

    # Detailed Inspection of Phase 2 Trades
    print("\n🔍 [Phase 2: 파워 아워 VWAP 스나이퍼 실제 발생 거래 전수 내역 (Case별)]")
    for r in results[1:]:
        p2_list = [t for t in r['trades'] if 'Phase 2' in t['phase']]
        print(f"\n--- [{r['case_id']}: {r['name']}] P2 거래 내역 (총 {len(p2_list)}건) ---")
        if p2_list:
            pdf = pd.DataFrame(p2_list)
            cols = ['date', 'symbol', 'entry_t', 'exit_t', 'entry_px', 'exit_px', 'ret_pct', 'net_pnl', 'exit_reason', 'holding_minutes']
            print(pdf[cols].to_string(index=False))
        else:
            print("   (발생한 Phase 2 거래 없음)")

    return results, summary_df

if __name__ == '__main__':
    run_power_hour_vwap_sniper_doe()
