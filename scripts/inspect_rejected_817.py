import sqlite3
import pandas as pd
import json
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
conn = sqlite3.connect(PROJECT_ROOT / "data" / "shadow_trades.db")
df = pd.read_sql_query("SELECT * FROM rejected_signals_simulation WHERE date_str = '2026-08-17' OR timestamp LIKE '2026-08-17%' ORDER BY timestamp ASC;", conn)
conn.close()

print(f"📊 [8/17 직전 장 미진입(차단) 신호 전수 조사: 총 {len(df)}건]")
print("=" * 85)

for idx, r in df.iterrows():
    print(f"[{idx+1}] 발생시각: {r['timestamp']} (NYT)")
    print(f"    • 지목 모델: {r['model_name']} ({r['expert_name']})")
    print(f"    • 당시 진입 후보가: ${r['candidate_price']:.2f} ({r['ticker']})")
    print(f"    • 당시 확신도 확률: {r['confidence_pct']}% (진입 승인 기준치: {r['threshold_pct']}%)")
    print(f"    • 미진입 차단 사유: {r['rejection_reason']}")
    print(f"    ---------------------------------------------------------------------")
    print(f"    🎯 [만약 당시 진입을 강행했다면 결과 (What-If 분석)]")
    print(f"    • 가상 청산 시점 및 가격: ${r['hypothetical_exit_price']:.2f} ({r['hypothetical_exit_reason']})")
    print(f"    • 수수료 차감 후 순손익: {r['hypothetical_pnl_pct']:+.2f}% (1000만원 기준 {r['hypothetical_pnl_krw']:+,}원)")
    print(f"    • AI 필터 판정: {r['filter_verdict']}")
    print("=" * 85)
