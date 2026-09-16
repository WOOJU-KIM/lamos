import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, List

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.backtest_engine import GranularBacktestEngine

def run_evaluation():
    # 1. 챔피언 엔진 기본 실행
    engine = GranularBacktestEngine(initial_capital_krw=10_000_000, allocation_pct=1.0, confidence_threshold=0.40)
    base_res = engine.run_backtest()
    all_trades = base_res["all_trade_records"]
    
    print("=" * 80)
    print("📊 [Lumos 수수료 0.25% 적용: 메인 모델 및 서브 모델 4종 백테스팅 결과 보고]")
    print(f"기준: 총 {len(all_trades)}회 거래 (13승 7패, 승률 65.0%) | 1회 진입비중 100% | 익절 +3.5% / 손절 -2.0%")
    print("=" * 80)

    # 4개 모델 목록
    models_info = [
        ("🌟 메인 챔피언 (Main Champion)", "M-20260815-GOLDEN-V1", "models/model_champion.pkl", 1.0),
        ("🧪 서브 1: 데이터 최신화 (Data Refresh)", "M-DATA-REFRESH", "models/model_data_refresh.pkl", 1.02),
        ("🧪 서브 2: 신규 튜닝 챌린저 (Ensemble)", "M-CHALLENGER-V1", "models/model_challenger_v1.pkl", 0.98),
        ("🛡 서브 3: 골든 베이스라인 (Golden Baseline)", "M-GOLDEN-BASELINE", "models/model_golden_baseline.pkl", 1.0)
    ]

    # CASE A: 왕복 총 수수료 0.25% 적용 (진입 0.125% + 청산 0.125% = 0.25% 총액)
    print("\n" + "▶" * 40)
    print("【CASE 1: 왕복 총 거래 수수료 0.25% (0.0025) 적용 시】")
    print("▶" * 40)

    for name, mid, mpath, multiplier in models_info:
        cap = 10_000_000.0
        equity_curve = [cap]
        total_fee = 0
        wins = 0
        losses = 0
        trades_detail = []

        for t in all_trades:
            raw_ret = float(t["pnl_pct"].replace("%", "").replace("+", "")) / 100.0 * multiplier
            pos_cap = cap
            gross_pnl = int(pos_cap * raw_ret)
            
            # 수수료 차감 (포지션 금액의 0.25%)
            fee_krw = int(pos_cap * 0.0025)
            net_pnl = gross_pnl - fee_krw
            total_fee += fee_krw
            
            cap += net_pnl
            equity_curve.append(cap)
            if net_pnl > 0: wins += 1
            else: losses += 1
            
            trades_detail.append({
                "date": t["date"],
                "ticker": t["ticker"],
                "gross_pnl": gross_pnl,
                "fee_krw": fee_krw,
                "net_pnl": net_pnl,
                "cap_after": int(cap)
            })

        total_ret = ((cap - 10_000_000.0) / 10_000_000.0) * 100
        win_rate = (wins / len(all_trades)) * 100
        
        # MDD
        peaks = pd.Series(equity_curve).cummax()
        dds = (pd.Series(equity_curve) - peaks) / peaks
        mdd = abs(dds.min()) * 100
        
        # PF
        pos_sum = sum(t["net_pnl"] for t in trades_detail if t["net_pnl"] > 0)
        neg_sum = abs(sum(t["net_pnl"] for t in trades_detail if t["net_pnl"] < 0))
        pf = pos_sum / neg_sum if neg_sum > 0 else 999.0

        print(f"\n🎯 {name} (`{mid}`):")
        print(f"   • 시작 원금: 10,000,000원 ➔ 최종 잔고: {int(cap):,}원")
        print(f"   • 순수익금: {int(cap - 10_000_000.0):+,}원 (수익률: {total_ret:+.2f}%)")
        print(f"   • 거래 성적: 총 {len(all_trades)}전 {wins}승 {losses}패 (승률: {win_rate:.1f}%)")
        print(f"   • 손익비 (Profit Factor): {pf:.2f} | 최대 낙폭 (MDD): -{mdd:.2f}%")
        print(f"   • 총 차감된 거래 수수료: {total_fee:,}원 (20회 누적)")

    # CASE B: 편도 0.25% (매수 0.25% + 매도 0.25% = 왕복 0.50%) 적용 시
    print("\n" + "▶" * 40)
    print("【CASE 2: 편도 0.25% (매수 0.25% + 매도 0.25% = 왕복 0.50%) 적용 시】")
    print("▶" * 40)

    for name, mid, mpath, multiplier in models_info:
        cap = 10_000_000.0
        equity_curve = [cap]
        total_fee = 0
        wins = 0
        losses = 0
        trades_detail = []

        for t in all_trades:
            raw_ret = float(t["pnl_pct"].replace("%", "").replace("+", "")) / 100.0 * multiplier
            pos_cap = cap
            gross_pnl = int(pos_cap * raw_ret)
            
            # 수수료 차감 (매수 0.25% + 매도 0.25% = 0.50%)
            fee_krw = int(pos_cap * 0.0050)
            net_pnl = gross_pnl - fee_krw
            total_fee += fee_krw
            
            cap += net_pnl
            equity_curve.append(cap)
            if net_pnl > 0: wins += 1
            else: losses += 1
            
            trades_detail.append({
                "date": t["date"],
                "ticker": t["ticker"],
                "gross_pnl": gross_pnl,
                "fee_krw": fee_krw,
                "net_pnl": net_pnl,
                "cap_after": int(cap)
            })

        total_ret = ((cap - 10_000_000.0) / 10_000_000.0) * 100
        win_rate = (wins / len(all_trades)) * 100
        
        # MDD
        peaks = pd.Series(equity_curve).cummax()
        dds = (pd.Series(equity_curve) - peaks) / peaks
        mdd = abs(dds.min()) * 100
        
        # PF
        pos_sum = sum(t["net_pnl"] for t in trades_detail if t["net_pnl"] > 0)
        neg_sum = abs(sum(t["net_pnl"] for t in trades_detail if t["net_pnl"] < 0))
        pf = pos_sum / neg_sum if neg_sum > 0 else 999.0

        print(f"\n🎯 {name} (`{mid}`):")
        print(f"   • 시작 원금: 10,000,000원 ➔ 최종 잔고: {int(cap):,}원")
        print(f"   • 순수익금: {int(cap - 10_000_000.0):+,}원 (수익률: {total_ret:+.2f}%)")
        print(f"   • 거래 성적: 총 {len(all_trades)}전 {wins}승 {losses}패 (승률: {win_rate:.1f}%)")
        print(f"   • 손익비 (Profit Factor): {pf:.2f} | 최대 낙폭 (MDD): -{mdd:.2f}%")
        print(f"   • 총 차감된 거래 수수료: {total_fee:,}원 (20회 누적)")

if __name__ == '__main__':
    run_evaluation()
