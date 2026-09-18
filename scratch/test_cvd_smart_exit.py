"""
================================================================================
Lumos V3: CVD 다이버전스 기반 조기 익절(Smart Exit) 섀도우 백테스트 엔진
================================================================================
- 메인 진입 로직: Lumos V3 (GBDT >= 55% + Cross-Asset Veto) 100% 동결 (절대 수정 없음)
- 청산 룰 비교:
  1) Baseline Model A: +3.5% TP / -2.0% SL / 90분 TimeStop / 당일 전량 청산
  2) Smart Exit Model B (Shadow): Baseline + 수익권(+1.5% 이상)에서 CVD 급락 다이버전스 발생 시 즉시 조기 익절
================================================================================
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List

# Windows 콘솔 UTF-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lumos")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator


# ==============================================================================
# 1. CVD 다이버전스 감지 함수
# ==============================================================================
def detect_cvd_divergence(price_series: pd.Series, cvd_series: pd.Series, window: int = 3) -> bool:
    """
    최근 N(window) 캔들 동안의 가격과 CVD 흐름을 비교 분석.
    - 가격 흐름: 상승 중이거나 고점에서 횡보 (Price Slope >= -0.0005)
    - CVD 흐름: 누적 델타가 급격히 꺾임 (CVD Slope < 0 & 순유출 음수)
    - 반환값: 매도 우위 다이버전스(조기 익절 트리거) 발생 시 True
    """
    if len(price_series) < window or len(cvd_series) < window:
        return False

    p_recent = price_series.tail(window).values
    c_recent = cvd_series.tail(window).values

    # 결측치 또는 0 방어
    if np.any(np.isnan(p_recent)) or np.any(np.isnan(c_recent)):
        return False

    x = np.arange(window)

    # 1. 가격 정규화 기울기 (Price Normalized Slope)
    p_mean = np.mean(p_recent) + 1e-9
    price_slope = np.polyfit(x, p_recent / p_mean, 1)[0]

    # 2. CVD 정규화 기울기 (CVD Normalized Slope)
    c_std = np.std(c_recent) + 1e-9
    cvd_slope = np.polyfit(x, (c_recent - np.mean(c_recent)) / c_std, 1)[0]

    # 3. 다이버전스 조건:
    # 가격은 유지되거나 상승(price_slope >= -0.001)하는데,
    # 대량 거래량 델타는 급격히 매도세로 전환(cvd_slope <= -0.5 및 최근봉 음수)
    is_price_holding_or_rising = (price_slope >= -0.001) or (p_recent[-1] >= p_recent[0])
    is_cvd_bleeding = (cvd_slope < -0.3) and (c_recent[-1] < c_recent[-2])

    return bool(is_price_holding_or_rising and is_cvd_bleeding)


# ==============================================================================
# 2. 섀도우 백테스트 엔진 (청산 로직 분기)
# ==============================================================================
def run_exit_shadow_backtest(
    df_market: pd.DataFrame,
    entry_signals: pd.Series,
    initial_capital: float = 10_000_000.0,
    smart_exit_min_profit: float = 0.015,  # 최소 +1.5% 이상 수익권일 때만 조기 익절 감시
    cvd_window: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    df_market: 15분봉 시세 및 CVD 지표가 포함된 통합 데이터프레임
    entry_signals: Lumos V3 (GBDT 55% + Cross-Asset Veto) 동일 진입 타점 ('TQQQ', 'SQQQ', 'HOLD')
    """
    TP_PCT = 0.035          # +3.5% 확정 익절
    SL_PCT = -0.020         # -2.0% 칼손절
    TIME_STOP_BARS = 6      # 90분 타임스탑 (15분봉 6개)
    SLIPPAGE_PAYUP = 0.03  # 주당 $0.03 슬리피지
    FEE_RATE = 0.0020       # 왕복 수수료 0.20%

    unique_dates = sorted(df_market['date'].unique())

    # --------------------------------------------------------------------------
    # 시뮬레이터 내부 공통 실행기
    # --------------------------------------------------------------------------
    def _simulate(enable_smart_exit: bool) -> List[Dict[str, Any]]:
        capital = float(initial_capital)
        trades = []
        active_pos = None

        for d_str in unique_dates:
            day_df = df_market[df_market['date'] == d_str]
            if len(day_df) < 5:
                continue

            for b_idx in range(len(day_df)):
                row = day_df.iloc[b_idx]
                dt = row.name
                time_str = row['time']

                # --------------------------------------------------------------
                # [A] 보유 포지션 청산 감시
                # --------------------------------------------------------------
                if active_pos is not None:
                    bars_held = b_idx - active_pos['entry_bar_idx'] + 1
                    sym = active_pos['symbol']
                    buy_px = active_pos['buy_price']

                    if sym == "TQQQ":
                        cur_h = row['high']
                        cur_l = row['low']
                        cur_c = row['close']
                        p_series = day_df['close'].iloc[max(0, b_idx - cvd_window + 1): b_idx + 1]
                        c_series = day_df['cvd'].iloc[max(0, b_idx - cvd_window + 1): b_idx + 1]
                    else:
                        cur_h = row['sqqq_high']
                        cur_l = row['sqqq_low']
                        cur_c = row['sqqq_close']
                        p_series = day_df['sqqq_close'].iloc[max(0, b_idx - cvd_window + 1): b_idx + 1]
                        c_series = day_df['sqqq_cvd'].iloc[max(0, b_idx - cvd_window + 1): b_idx + 1]

                    max_ret = (cur_h - buy_px) / buy_px
                    min_ret = (cur_l - buy_px) / buy_px
                    cur_ret = (cur_c - buy_px) / buy_px

                    exit_triggered = False
                    exit_reason = ""
                    exit_px = 0.0

                    # 1. 목표가 (+3.5%) 확정 익절
                    if max_ret >= TP_PCT:
                        exit_triggered = True
                        exit_reason = "TP (+3.5%)"
                        exit_px = round(buy_px * (1 + TP_PCT) - SLIPPAGE_PAYUP, 2)

                    # 2. 칼손절 (-2.0%)
                    elif min_ret <= SL_PCT:
                        exit_triggered = True
                        exit_reason = "SL (-2.0%)"
                        exit_px = round(buy_px * (1 + SL_PCT) - SLIPPAGE_PAYUP, 2)

                    # 3. [신규 섀도우 룰] CVD 다이버전스 조기 익절 (Smart Exit)
                    #    - 수익권(+1.5% 이상)에 도달해 있는데,
                    #    - 가격 상승 대비 CVD가 급락하는 매도 다이버전스 감지 시 선제 청산
                    elif enable_smart_exit and (cur_ret >= smart_exit_min_profit):
                        if detect_cvd_divergence(p_series, c_series, window=cvd_window):
                            exit_triggered = True
                            exit_reason = f"CVD Smart Exit (+{cur_ret*100:.2f}%)"
                            exit_px = round(cur_c - SLIPPAGE_PAYUP, 2)

                    # 4. 90분 타임스탑
                    elif bars_held >= TIME_STOP_BARS:
                        exit_triggered = True
                        exit_reason = "TimeStop (90m)"
                        exit_px = round(cur_c - SLIPPAGE_PAYUP, 2)

                    # 5. 당일 정규장 마감 청산 (15:45~15:50 NYT)
                    elif b_idx >= len(day_df) - 1 or time_str >= "15:45":
                        exit_triggered = True
                        exit_reason = "EOD Close"
                        exit_px = round(cur_c - SLIPPAGE_PAYUP, 2)

                    if exit_triggered:
                        invested = active_pos['invested']
                        shares = active_pos['shares']
                        cost_amount = invested * FEE_RATE
                        pnl_amount = (shares * (exit_px - buy_px)) - cost_amount
                        pnl_pct = (exit_px - buy_px) / buy_px - FEE_RATE
                        capital += pnl_amount

                        trades.append({
                            "symbol": sym,
                            "entry_dt": active_pos['entry_dt'],
                            "exit_dt": dt,
                            "entry_px": buy_px,
                            "exit_px": exit_px,
                            "pnl_amount": pnl_amount,
                            "pnl_pct": pnl_pct,
                            "is_win": pnl_amount > 0,
                            "exit_reason": exit_reason,
                            "bars_held": bars_held,
                            "capital_after": capital
                        })
                        active_pos = None
                        continue

                # --------------------------------------------------------------
                # [B] 신규 진입 (Lumos V3 메인 로직 100% 동일)
                # --------------------------------------------------------------
                if active_pos is None:
                    if time_str < "09:45" or time_str >= "14:30":
                        continue

                    sig = entry_signals.loc[dt]
                    if sig not in ['TQQQ', 'SQQQ']:
                        continue

                    # 3중 스크린 검증
                    if sig == 'TQQQ':
                        if not (row['trend_ok_tqqq'] and row['dip_ok_tqqq']):
                            continue
                        cur_px = row['close']
                    else:
                        if not (row['trend_ok_sqqq'] and row['dip_ok_sqqq']):
                            continue
                        cur_px = row['sqqq_close']

                    entry_order_px = round(cur_px + SLIPPAGE_PAYUP, 2)
                    invest_amt = capital * 0.98
                    shares = int(invest_amt / entry_order_px) if entry_order_px > 0 else 0

                    if shares > 0:
                        active_pos = {
                            "symbol": sig,
                            "buy_price": entry_order_px,
                            "shares": shares,
                            "invested": shares * entry_order_px,
                            "entry_dt": dt,
                            "entry_bar_idx": b_idx
                        }

        return trades

    # 1. Baseline 시뮬레이션 (enable_smart_exit=False)
    trades_base = _simulate(enable_smart_exit=False)
    # 2. Smart Exit 시뮬레이션 (enable_smart_exit=True)
    trades_smart = _simulate(enable_smart_exit=True)

    df_base = pd.DataFrame(trades_base)
    df_smart = pd.DataFrame(trades_smart)

    return df_base, df_smart, {"initial_capital": initial_capital}


# ==============================================================================
# 3. 성과 비교 리포트 출력 함수
# ==============================================================================
def print_exit_comparison(df_base: pd.DataFrame, df_smart: pd.DataFrame, initial_capital: float = 10_000_000.0):
    """
    기존 청산(Baseline)과 CVD 조기 익절(Smart Exit) 간의 정밀 퀀트 지표 맞대결 리포트
    """
    def _metrics(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {
                "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "profit_factor": 0.0, "total_return": 0.0, "final_cap": initial_capital,
                "mdd": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "smart_exit_count": 0
            }

        n = len(df)
        wins = int(df['is_win'].sum())
        losses = n - wins
        wr = (wins / n) * 100.0 if n > 0 else 0.0

        pos_pnl = df[df['pnl_amount'] > 0]['pnl_amount'].sum()
        neg_pnl = abs(df[df['pnl_amount'] < 0]['pnl_amount'].sum())
        pf = (pos_pnl / (neg_pnl + 1e-9)) if neg_pnl > 0 else (99.9 if pos_pnl > 0 else 1.0)

        final_cap = df['capital_after'].iloc[-1]
        tot_ret = ((final_cap - initial_capital) / initial_capital) * 100.0

        # MDD 산출
        cum_caps = pd.Series([initial_capital] + df['capital_after'].tolist())
        peaks = cum_caps.cummax()
        dds = (peaks - cum_caps) / peaks * 100.0
        mdd = dds.max()

        avg_w = df[df['pnl_amount'] > 0]['pnl_pct'].mean() * 100.0 if wins > 0 else 0.0
        avg_l = abs(df[df['pnl_amount'] < 0]['pnl_pct'].mean() * 100.0) if losses > 0 else 0.0
        smart_cnt = int(df['exit_reason'].str.contains('CVD Smart Exit').sum())

        return {
            "trades": n, "wins": wins, "losses": losses, "win_rate": wr,
            "profit_factor": pf, "total_return": tot_ret, "final_cap": final_cap,
            "mdd": mdd, "avg_win": avg_w, "avg_loss": avg_l, "smart_exit_count": smart_cnt
        }

    m_base = _metrics(df_base)
    m_smart = _metrics(df_smart)

    print("\n" + "=" * 85)
    print("🏛 [Lumos V3 청산 알고리즘 섀도우 백테스트 비교 성적표]")
    print("=" * 85)
    print(f"{'성과 지표':<26} | {'Model A (기존 4대 청산)':<24} | {'Model B (CVD Smart Exit 추가)':<26} | {'차이 (Delta)':<12}")
    print("-" * 85)

    diff_trades = m_smart['trades'] - m_base['trades']
    diff_wr = m_smart['win_rate'] - m_base['win_rate']
    diff_pf = m_smart['profit_factor'] - m_base['profit_factor']
    diff_ret = m_smart['total_return'] - m_base['total_return']
    diff_mdd = m_smart['mdd'] - m_base['mdd']
    diff_cap = m_smart['final_cap'] - m_base['final_cap']

    print(f"{'총 거래 횟수':<26} | {m_base['trades']:>4d}회 ({m_base['wins']}승 {m_base['losses']}패)        | {m_smart['trades']:>4d}회 ({m_smart['wins']}승 {m_smart['losses']}패)          | {diff_trades:>+4d}회")
    print(f"{'백테스트 승률 (Win Rate)':<24} | {m_base['win_rate']:>20.2f}% | {m_smart['win_rate']:>22.2f}% | {diff_wr:>+7.2f}%p")
    print(f"{'손익비 (Profit Factor)':<24} | {m_base['profit_factor']:>21.2f} | {m_smart['profit_factor']:>23.2f} | {diff_pf:>+8.2f}")
    print(f"{'누적 수익률 (Total Return)':<23} | {m_base['total_return']:>19.2f}% | {m_smart['total_return']:>21.2f}% | {diff_ret:>+7.2f}%p")
    print(f"{'최대 낙폭 (MDD)':<26} | {m_base['mdd']:>20.2f}% | {m_smart['mdd']:>22.2f}% | {diff_mdd:>+7.2f}%p")
    print(f"{'최종 자본금 (Final Capital)':<23} | {int(m_base['final_cap']):>18,d}원 | {int(m_smart['final_cap']):>20,d}원 | {int(diff_cap):>+10,d}원")
    print(f"{'평균 익절률 (Avg Win)':<25} | {m_base['avg_win']:>20.2f}% | {m_smart['avg_win']:>22.2f}% | {m_smart['avg_win']-m_base['avg_win']:>+7.2f}%p")
    print(f"{'평균 손절률 (Avg Loss)':<24} | {m_base['avg_loss']:>20.2f}% | {m_smart['avg_loss']:>22.2f}% | {m_smart['avg_loss']-m_base['avg_loss']:>+7.2f}%p")
    print(f"{'CVD 조기 익절 발동 횟수':<23} | {'해당 없음 (0회)':>21} | {m_smart['smart_exit_count']:>19d}회 | {m_smart['smart_exit_count']:>+8d}회")
    print("=" * 85)

    # 4. 청산 사유별 상세 분포 분석
    print("\n📊 [청산 사유별 상세 분포 비교]")
    print(f"{'청산 사유':<26} | {'Model A 건수':<15} | {'Model B 건수':<15}")
    print("-" * 65)
    reasons = set(df_base['exit_reason'].unique()).union(set(df_smart['exit_reason'].unique()))
    for r in sorted(reasons):
        c_a = int((df_base['exit_reason'] == r).sum())
        c_b = int((df_smart['exit_reason'] == r).sum())
        print(f"{r:<28} | {c_a:>12d}건 | {c_b:>12d}건")
    print("=" * 65)


# ==============================================================================
# 4. 통합 실행 파이프라인
# ==============================================================================
def run_comparison_pipeline():
    print("=" * 85)
    print("🏛 [Lumos V3 Execution 퀀트 리서치: CVD Smart Exit 섀도우 모델 검증]")
    print("=" * 85)
    print("⏳ [1/3] 로컬 데이터레이크(market_data.db)로부터 15분봉 및 CVD 데이터 로드 중...")

    lake = MarketDataLake()
    tqqq_15m = lake.load_candles('TQQQ', '15m')
    sqqq_15m = lake.load_candles('SQQQ', '15m')
    nvda_15m = lake.load_candles('NVDA', '15m')
    qqq_15m = lake.load_candles('QQQ', '15m')
    vix_15m = lake.load_candles('^VIX', '15m')
    soxx_15m = lake.load_candles('SOXX', '15m')
    soxx_60m = lake.load_candles('SOXX', '60m')
    tqqq_60m = lake.load_candles('TQQQ', '60m')

    moe = MoEMetaOrchestrator()
    gbdt_engine = moe.gbdt_engine
    cross_model = moe.cross_asset_model

    # GBDT Features
    tqqq_feat = gbdt_engine.extract_features(tqqq_15m)
    tqqq_feat = gbdt_engine.add_confidence_columns(tqqq_feat)

    common_idx = tqqq_15m.index.intersection(nvda_15m.index).intersection(qqq_15m.index).intersection(vix_15m.index)

    # CVD (Cumulative Volume Delta) 계산
    # Delta = Volume * (2*Close - High - Low) / (High - Low + 1e-6)
    def calc_cvd(df_c):
        h = df_c['High']
        l = df_c['Low']
        c = df_c['Close']
        v = df_c['Volume']
        delta = v * (2.0 * c - h - l) / (h - l + 1e-6)
        # 당일 리셋 CVD (Intraday CVD)
        date_series = df_c.index.strftime('%Y-%m-%d')
        df_tmp = pd.DataFrame({'delta': delta, 'date': date_series}, index=df_c.index)
        return df_tmp.groupby('date')['delta'].cumsum()

    cvd_tqqq = calc_cvd(tqqq_15m.loc[common_idx])
    cvd_sqqq = calc_cvd(sqqq_15m.loc[common_idx])

    # Cross-Asset 계산
    tqqq_ret5 = (tqqq_15m.loc[common_idx, 'Close'] / tqqq_15m.loc[common_idx, 'Close'].shift(5) - 1.0).fillna(0.0)
    nvda_ret5 = (nvda_15m.loc[common_idx, 'Close'] / nvda_15m.loc[common_idx, 'Close'].shift(5) - 1.0).fillna(0.0)
    qqq_ret5 = (qqq_15m.loc[common_idx, 'Close'] / qqq_15m.loc[common_idx, 'Close'].shift(5) - 1.0).fillna(0.0)
    vix_ret5 = (vix_15m.loc[common_idx, 'Close'] / vix_15m.loc[common_idx, 'Close'].shift(5) - 1.0).fillna(0.0)
    soxx_ret5 = (soxx_15m.loc[common_idx, 'Close'] / soxx_15m.loc[common_idx, 'Close'].shift(5) - 1.0).fillna(0.0) if not soxx_15m.empty else tqqq_ret5

    cross_dirs = []
    for dt in common_idx:
        sig_c, _, _ = cross_model.predict_signal(
            tqqq_ret=tqqq_ret5.loc[dt],
            nvda_ret=nvda_ret5.loc[dt],
            qqq_ret=qqq_ret5.loc[dt],
            soxx_ret=soxx_ret5.loc[dt],
            vix_ret=vix_ret5.loc[dt],
            tnx_ret=0.0
        )
        if sig_c > 0:
            cross_dirs.append('TQQQ')
        elif sig_c < 0:
            cross_dirs.append('SQQQ')
        else:
            cross_dirs.append('HOLD')

    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
    tqqq_60m['ema20'] = tqqq_60m['Close'].ewm(span=20, adjust=False).mean()

    df_market = pd.DataFrame(index=common_idx)
    df_market['open'] = tqqq_15m.loc[common_idx, 'Open']
    df_market['high'] = tqqq_15m.loc[common_idx, 'High']
    df_market['low'] = tqqq_15m.loc[common_idx, 'Low']
    df_market['close'] = tqqq_15m.loc[common_idx, 'Close']
    df_market['volume'] = tqqq_15m.loc[common_idx, 'Volume']
    df_market['cvd'] = cvd_tqqq

    df_market['sqqq_open'] = sqqq_15m.loc[common_idx, 'Open']
    df_market['sqqq_high'] = sqqq_15m.loc[common_idx, 'High']
    df_market['sqqq_low'] = sqqq_15m.loc[common_idx, 'Low']
    df_market['sqqq_close'] = sqqq_15m.loc[common_idx, 'Close']
    df_market['sqqq_volume'] = sqqq_15m.loc[common_idx, 'Volume']
    df_market['sqqq_cvd'] = cvd_sqqq

    # 60m 추세 필터 (Screen 1)
    trend_ok_tqqq, trend_ok_sqqq = [], []
    for dt in common_idx:
        p_soxx = soxx_60m[soxx_60m.index <= dt]
        p_tqqq = tqqq_60m[tqqq_60m.index <= dt]
        if len(p_soxx) >= 20 and len(p_tqqq) >= 20:
            soxx_c = p_soxx['Close'].iloc[-1]
            soxx_e20 = p_soxx['ema20'].iloc[-1]
            tqqq_c = p_tqqq['Close'].iloc[-1]
            tqqq_e20 = p_tqqq['ema20'].iloc[-1]
            trend_ok_tqqq.append((soxx_c >= soxx_e20 * 0.998) and (tqqq_c >= tqqq_e20 * 0.998))
            trend_ok_sqqq.append(soxx_c <= soxx_e20 * 1.002)
        else:
            trend_ok_tqqq.append(True)
            trend_ok_sqqq.append(True)

    df_market['trend_ok_tqqq'] = trend_ok_tqqq
    df_market['trend_ok_sqqq'] = trend_ok_sqqq

    # GBDT 및 Cross-Asset
    c_feat = tqqq_feat.loc[common_idx]
    gbdt_sig = c_feat['Signal']
    gbdt_conf = c_feat['Confidence']
    df_market['gbdt_dir'] = np.where(gbdt_sig == 1, 'TQQQ', np.where(gbdt_sig == -1, 'SQQQ', 'HOLD'))
    df_market['gbdt_prob'] = gbdt_conf.values
    df_market['cross_dir'] = cross_dirs

    # Screen 3 columns
    df_market['dip_ok_tqqq'] = (c_feat['VWAP_Diff'] <= 1.5) & (c_feat['RSI_14'] <= 62.0)
    df_market['dip_ok_sqqq'] = (c_feat['VWAP_Diff'] >= -1.5) & (c_feat['RSI_14'] >= 38.0)

    df_market['date'] = df_market.index.strftime('%Y-%m-%d')
    df_market['time'] = df_market.index.strftime('%H:%M')

    # Lumos V3 메인 진입 시그널 산출 (GBDT >= 55% + Cross-Asset Veto)
    print("⏳ [2/3] Lumos V3 불변 메인 진입 타점 고정 추출 (GBDT >= 55% + Cross-Asset Opposite Veto)...")
    is_gbdt_tqqq = (df_market['gbdt_dir'] == 'TQQQ') & (df_market['gbdt_prob'] >= 0.55)
    is_gbdt_sqqq = (df_market['gbdt_dir'] == 'SQQQ') & (df_market['gbdt_prob'] >= 0.55)
    veto_tqqq = is_gbdt_tqqq & (df_market['cross_dir'] == 'SQQQ')
    veto_sqqq = is_gbdt_sqqq & (df_market['cross_dir'] == 'TQQQ')

    entry_conds = [
        is_gbdt_tqqq & (~veto_tqqq),
        is_gbdt_sqqq & (~veto_sqqq)
    ]
    entry_signals = pd.Series(np.select(entry_conds, ['TQQQ', 'SQQQ'], default='HOLD'), index=df_market.index)

    # 섀도우 백테스트 실행
    print("⏳ [3/3] Baseline(기존 4대 청산) vs Shadow(CVD Smart Exit) 병렬 백테스트 시뮬레이션 집행...")
    df_base, df_smart, meta = run_exit_shadow_backtest(
        df_market=df_market,
        entry_signals=entry_signals,
        initial_capital=10_000_000.0,
        smart_exit_min_profit=0.015,  # +1.5% 이상 수익권 진입 시 CVD 다이버전스 감시
        cvd_window=3
    )

    print_exit_comparison(df_base, df_smart, initial_capital=10_000_000.0)

if __name__ == "__main__":
    run_comparison_pipeline()
