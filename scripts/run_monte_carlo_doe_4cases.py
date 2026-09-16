import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_doe_confidence_sweep import run_doe_confidence_sweep

def run_doe_monte_carlo():
    all_results, trades_dict = run_doe_confidence_sweep()
    
    # T = 60% (Lumos V3 하드 룰 인터락 기준) 거래 목록
    trades_60 = trades_dict.get("60%", [])
    if not trades_60:
        raise ValueError("T=60% 거래 데이터를 찾을 수 없습니다.")

    df_trades = pd.DataFrame(trades_60)
    print("\n" + "=" * 95)
    print(f"📊 [DoE T=60% 실제 거래 표본 추출 완료: 총 {len(df_trades)}회]")
    print("=" * 95)
    
    # return_pct 추출 (/ 100.0)
    pnl = df_trades['return_pct'].values / 100.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    
    actual_wr = len(wins) / len(pnl)
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    payoff = abs(avg_win / avg_loss)
    
    print(f"• 표본 거래 수: {len(df_trades)}회 (34승 12패)")
    print(f"• 현재 DoE 실전 승률: {actual_wr * 100:.2f}%")
    print(f"• 평균 익절률: {avg_win * 100:+.2f}% | 평균 손실률: {avg_loss * 100:+.2f}%")
    print(f"• 손익비 (Payoff Ratio): {payoff:.2f}")

    n_simulations = 20000
    n_trades = 100
    initial_capital = 10_000_000.0

    scenarios = [
        {"name": f"1. 현재 실전 모델 승률 ({actual_wr*100:.1f}%)", "win_rate": actual_wr, "tag": f"Current ({actual_wr*100:.1f}%)"},
        {"name": "2. 승률 60% 보수적 시나리오 (60.0%)", "win_rate": 0.60, "tag": "60.0%"},
        {"name": "3. 승률 50% 손익분기 방어선 (50.0%)", "win_rate": 0.50, "tag": "50.0%"},
        {"name": "4. 승률 40% 최악의 역풍 시나리오 (40.0%)", "win_rate": 0.40, "tag": "40.0%"},
    ]

    np.random.seed(42)
    summary = []

    for sc in scenarios:
        wr = sc["win_rate"]
        name = sc["name"]

        is_win = np.random.rand(n_simulations, n_trades) < wr
        win_returns = np.random.choice(wins, size=(n_simulations, n_trades), replace=True)
        loss_returns = np.random.choice(losses, size=(n_simulations, n_trades), replace=True)

        returns = np.where(is_win, win_returns, loss_returns)

        # 복리 자산 곡선 (100% 자본 투입)
        multipliers = 1.0 + returns
        cum_multipliers = np.cumprod(multipliers, axis=1)
        capital_curves = np.hstack([np.ones((n_simulations, 1)), cum_multipliers]) * initial_capital

        final_capitals = capital_curves[:, -1]
        total_returns = (final_capitals - initial_capital) / initial_capital * 100.0

        # Drawdown
        running_max = np.maximum.accumulate(capital_curves, axis=1)
        drawdowns = (running_max - capital_curves) / running_max * 100.0
        max_drawdowns = np.max(drawdowns, axis=1)

        # 파산 확률 (MDD 임계치)
        prob_dd_15 = np.mean(max_drawdowns >= 15.0) * 100.0
        prob_dd_20 = np.mean(max_drawdowns >= 20.0) * 100.0
        prob_dd_30 = np.mean(max_drawdowns >= 30.0) * 100.0
        prob_dd_50 = np.mean(max_drawdowns >= 50.0) * 100.0
        prob_dd_70 = np.mean(max_drawdowns >= 70.0) * 100.0
        prob_loss_finish = np.mean(final_capitals < initial_capital) * 100.0

        # 최대 연속 손실 계산
        is_loss = ~is_win
        max_consec_losses = []
        for i in range(n_simulations):
            row = is_loss[i]
            max_c = 0
            cur_c = 0
            for val in row:
                if val:
                    cur_c += 1
                    if cur_c > max_c:
                        max_c = cur_c
                else:
                    cur_c = 0
            max_consec_losses.append(max_c)
        max_consec_losses = np.array(max_consec_losses)

        stat = {
            "scenario": name,
            "win_rate_pct": wr * 100.0,
            "expected_return_pct": np.mean(total_returns),
            "median_return_pct": np.median(total_returns),
            "worst_return_pct": np.min(total_returns),
            "p01_return_pct": np.percentile(total_returns, 1),
            "p05_return_pct": np.percentile(total_returns, 5),
            "p25_return_pct": np.percentile(total_returns, 25),
            "p75_return_pct": np.percentile(total_returns, 75),
            "p95_return_pct": np.percentile(total_returns, 95),
            "p99_return_pct": np.percentile(total_returns, 99),
            "best_return_pct": np.max(total_returns),
            "mean_mdd_pct": np.mean(max_drawdowns),
            "median_mdd_pct": np.median(max_drawdowns),
            "p95_mdd_pct": np.percentile(max_drawdowns, 95),
            "p99_mdd_pct": np.percentile(max_drawdowns, 99),
            "worst_mdd_pct": np.max(max_drawdowns),
            "prob_dd_15_pct": prob_dd_15,
            "prob_dd_20_pct": prob_dd_20,
            "prob_dd_30_pct": prob_dd_30,
            "prob_dd_50_pct": prob_dd_50,
            "prob_dd_70_pct": prob_dd_70,
            "prob_loss_finish_pct": prob_loss_finish,
            "avg_max_consec_loss": np.mean(max_consec_losses),
            "p95_consec_loss": np.percentile(max_consec_losses, 95),
            "p99_consec_loss": np.percentile(max_consec_losses, 99),
            "worst_consec_loss": np.max(max_consec_losses),
            "median_final_capital": np.median(final_capitals),
            "worst_final_capital": np.min(final_capitals),
            "p01_final_capital": np.percentile(final_capitals, 1),
            "p05_final_capital": np.percentile(final_capitals, 5),
            "p95_final_capital": np.percentile(final_capitals, 95),
        }
        summary.append(stat)

    df_summary = pd.DataFrame(summary)
    
    print("\n" + "=" * 95)
    print("🎲 [DoE 기반 몬테카를로 20,000회 시뮬레이션 최종 결산]")
    print("=" * 95)
    for idx, r in df_summary.iterrows():
        print(f"\n▶ [{r['scenario']}]")
        print(f"  • 수익률 분포:")
        print(f"    - 기대(평균) 수익률: {r['expected_return_pct']:+.2f}% | 중앙값: {r['median_return_pct']:+.2f}%")
        print(f"    - 최악 경로 (2만번 중 1등): {r['worst_return_pct']:+.2f}% (잔고: {int(r['worst_final_capital']):,}원)")
        print(f"    - 하위 1% 극단 구간 (99% VaR): {r['p01_return_pct']:+.2f}% (잔고: {int(r['p01_final_capital']):,}원)")
        print(f"    - 하위 5% 비관 구간 (95% VaR): {r['p05_return_pct']:+.2f}% (잔고: {int(r['p05_final_capital']):,}원)")
        print(f"    - 상위 5% 최상 구간: {r['p95_return_pct']:+.2f}% (잔고: {int(r['p95_final_capital']):,}원)")
        print(f"  • 최대 낙폭 (MDD):")
        print(f"    - 평균 MDD: {r['mean_mdd_pct']:.2f}% | 95% 신뢰수준 MDD: {r['p95_mdd_pct']:.2f}% | 최악 MDD: {r['worst_mdd_pct']:.2f}%")
        print(f"  • 연속 손실:")
        print(f"    - 평균 {r['avg_max_consec_loss']:.1f}연패 | 95% {int(r['p95_consec_loss'])}연패 | 최악 {int(r['worst_consec_loss'])}연패")
        print(f"  • 파산 확률 (Risk of Ruin):")
        print(f"    - 경고선 (MDD ≥ 15%): {r['prob_dd_15_pct']:.2f}%")
        print(f"    - 주의선 (MDD ≥ 20%): {r['prob_dd_20_pct']:.2f}%")
        print(f"    - 위험선 (MDD ≥ 30%): {r['prob_dd_30_pct']:.2f}%")
        print(f"    - 반토막 폐기선 (MDD ≥ 50%): {r['prob_dd_50_pct']:.2f}%")
        print(f"    - 깡통 위험선 (MDD ≥ 70%): {r['prob_dd_70_pct']:.2f}%")
        print(f"    - 100회 후 원금 손실 마감 확률: {r['prob_loss_finish_pct']:.2f}%")

    return df_summary

if __name__ == "__main__":
    run_doe_monte_carlo()
