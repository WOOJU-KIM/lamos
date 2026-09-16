import json
from pathlib import Path
from collections import defaultdict

log_file = Path("data/system_logs.jsonl")

trades = []
buy_events = []
sell_events = []
carry_events = []

if log_file.exists():
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line.strip())
                ts = data.get("timestamp", "")
                cat = data.get("category", "")
                action = data.get("action", "")
                msg = data.get("message", "")

                # 2026-08-24 장 시작(22:30 KST) 이후 로그만 필터링
                if "2026-08-24 22:" in ts or "2026-08-25 0" in ts or "2026-08-25 04:" in ts:
                    if action in ["AutoExecution", "CarryOverClear", "TakeProfit", "StopLoss", "TimeStop", "EODLiquidation", "CircuitBreaker"]:
                        trades.append({
                            "timestamp": ts,
                            "action": action,
                            "message": msg
                        })
            except Exception as e:
                pass

print(f"=== 어제 정규장(22:30 ~ 05:00) 총 매매 이벤트: {len(trades)}건 ===")

action_counts = defaultdict(int)
for t in trades:
    action_counts[t["action"]] += 1

print("\n--- 이벤트 종류별 통계 ---")
for k, v in action_counts.items():
    print(f"• {k}: {v}건")

print("\n--- 상세 내역 샘플 (최초 10건 & 마지막 10건) ---")
for t in trades[:10]:
    print(f"[{t['timestamp']}][{t['action']}] {t['message']}")
print("...")
for t in trades[-10:]:
    print(f"[{t['timestamp']}][{t['action']}] {t['message']}")
