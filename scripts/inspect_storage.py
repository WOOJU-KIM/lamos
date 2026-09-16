import sys
import os
import sqlite3
import pandas as pd
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

DATA_DIR = PROJECT_ROOT / "data"

def inspect_storage():
    print("=" * 75)
    print("📦 [통합 저장소 점검: SQLite DB & CSV & JSON 이력 보존 상태] 📦")
    print("=" * 75)

    print("\n1️⃣ [data/ 디렉토리 저장 파일 목록]")
    for f in sorted(os.listdir(DATA_DIR)):
        p = DATA_DIR / f
        if p.is_file():
            print(f"  • data/{f:<25} ({p.stat().st_size:,} bytes)")

    db_path = DATA_DIR / "system_hub.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n2️⃣ [SQLite Database (data/system_hub.db) 테이블별 건수]")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    for t in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cursor.fetchone()[0]
        print(f"  • 테이블 [{t:<18}]: 총 {cnt:>4}건 영구 보존 중")

    print("\n3️⃣ [backtest_runs] 최근 백테스트 실행 이력 (최근 3건)")
    df_runs = pd.read_sql(
        "SELECT run_id, executed_at, total_trades, wins, losses, win_rate_pct, total_return_pct, final_capital_krw, profit_factor, mdd_pct FROM backtest_runs ORDER BY rowid DESC LIMIT 3",
        conn
    )
    print(df_runs.to_string(index=False))

    print("\n4️⃣ [trades] 최근 체결된 개별 거래 내역 (최근 5건)")
    df_trades = pd.read_sql(
        "SELECT trade_id, date, ticker, entry_price, exit_price, pnl_pct, pnl_krw, exit_reason FROM trades ORDER BY rowid DESC LIMIT 5",
        conn
    )
    print(df_trades.to_string(index=False))

    print("\n5️⃣ [JSON Summary (data/trade_logs_summary.json) 메타데이터 요약]")
    summary_path = DATA_DIR / "trade_logs_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            print(f"  • 총 기록 거래 수: {meta.get('total_trades_logged')}건")
            print(f"  • 최종 잔고: {meta.get('final_capital_krw'):,}원 ({meta.get('total_return_pct'):+.2f}%)")
            print(f"  • 승률: {meta.get('win_rate_pct')}% | MDD: -{meta.get('mdd_pct')}% | PF: {meta.get('profit_factor')}")
            print(f"  • Top 3 핵심 피처: {meta.get('top_3_features')}")
            print(f"  • 최종 저장 일시: {meta.get('last_updated_at')}")

    conn.close()
    print("\n" + "=" * 75)

if __name__ == "__main__":
    inspect_storage()
