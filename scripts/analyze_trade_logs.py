import json
from pathlib import Path
from collections import defaultdict

log_file = Path("data/system_logs.jsonl")

trades = []
all_categories = defaultdict(int)
all_actions = defaultdict(int)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            d = json.loads(line.strip())
            ts = d.get("timestamp", "")
            cat = d.get("category", "")
            act = d.get("action", "")
            msg = d.get("message", "")
            
            all_categories[cat] += 1
            all_actions[act] += 1
            
            # 매매 관련 액션
            if cat in ["TRADE", "ORDER", "POSITION"] or act in ["AutoExecution", "CarryOverClear", "TakeProfit", "StopLoss", "TimeStop", "EODLiquidation", "CircuitBreaker"]:
                trades.append(d)
        except Exception:
            pass

summary = {
    "total_trade_events": len(trades),
    "action_counts": dict(all_actions),
    "category_counts": dict(all_categories),
    "sample_first_20": trades[:20],
    "sample_last_20": trades[-20:]
}

with open("data/trade_logs_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Total trade events: {len(trades)}")
print("Action counts:", dict(all_actions))
