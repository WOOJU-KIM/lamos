import json
from collections import defaultdict

trade_events = []
tp_list = []
sl_list = []
carry_list = []
timestop_list = []
buy_list = []
eod_list = []

with open('data/system_logs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        try:
            d = json.loads(line.strip())
            dt = d.get("datetime", "")
            msg = d.get("message", "")
            src = d.get("source", "")
            lvl = d.get("level", "")

            # 2026-08-24 22:30 이후부터 2026-08-25 05:00까지
            if ("2026-08-24 22:" in dt or "2026-08-24 23:" in dt or 
                "2026-08-25 00:" in dt or "2026-08-25 01:" in dt or 
                "2026-08-25 02:" in dt or "2026-08-25 03:" in dt or 
                "2026-08-25 04:" in dt):
                
                if "매수" in msg or "BUY" in msg or "익절" in msg or "손절" in msg or "청산" in msg or "CarryOverClear" in src or "TakeProfit" in src or "StopLoss" in src:
                    trade_events.append(d)
                    
                    if "TakeProfit" in src or "익절" in msg:
                        tp_list.append(d)
                    elif "StopLoss" in src or "손절" in msg:
                        sl_list.append(d)
                    elif "CarryOverClear" in src or "미청산 잔여분" in msg or "개장" in msg:
                        carry_list.append(d)
                    elif "TimeStop" in src or "타임스탑" in msg:
                        timestop_list.append(d)
                    elif "장마감" in msg or "EODLiquidation" in src or "0% 청산" in msg:
                        eod_list.append(d)
                    elif "매수" in msg or "AutoExecution" in src:
                        buy_list.append(d)
        except Exception:
            pass

res = {
    "total_trade_related_logs": len(trade_events),
    "counts": {
        "buy_orders (매수 발주)": len(buy_list),
        "take_profit (목표 익절 +3.5%)": len(tp_list),
        "stop_loss (칼손절 -2.0%)": len(sl_list),
        "carry_over_clear (사자마자 개장청산 버그)": len(carry_list),
        "timestop (90분 타임스탑)": len(timestop_list),
        "eod_liquidation (장마감 전량청산)": len(eod_list)
    },
    "samples": {
        "first_5_buys": buy_list[:5],
        "first_5_carry_clears": carry_list[:5],
        "all_take_profits": tp_list,
        "all_stop_losses": sl_list
    }
}

with open("data/trade_logs_summary.json", "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)

print("Analysis completed and saved to data/trade_logs_summary.json")
