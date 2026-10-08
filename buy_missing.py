import time
import sqlite3
import yfinance as yf
from trader import KiwoomRESTTrader

def buy_missing():
    trader = KiwoomRESTTrader()
    
    conn = sqlite3.connect("quant_system.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM daily_candidates WHERE date = date('now') ORDER BY rank ASC LIMIT 15")
    candidates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    pf = trader.get_portfolio()
    current_holdings = [p["symbol"] for p in pf]
    
    # 아까 62,242.76 / 15 로 계산된 1종목당 목표 금액
    target_amount = 4149.51
    
    print("=== 누락된 종목 매수 ===")
    for tk in candidates:
        if tk in current_holdings:
            print(f"[SKIP] {tk} 이미 보유 중")
            continue
            
        retry_count = 0
        while True:
            try:
                curr_px = yf.Ticker(tk).fast_info.last_price
            except:
                time.sleep(1)
                continue
                
            order_price = round(curr_px + 0.05, 2)
            qty = int(target_amount // order_price)
            
            if qty <= 0:
                print(f"[PASS] {tk} 매수 금액 부족")
                break
                
            print(f"[매수] {tk} {qty}주 @ ${order_price:.2f} (현재가 +0.05)")
            trader.send_order(tk, "BUY", qty, order_price)
            
            filled = False
            for _ in range(10):
                time.sleep(1)
                pf_check = trader.get_portfolio()
                tk_holding = next((x for x in pf_check if x["symbol"] == tk), None)
                if tk_holding and tk_holding["qty"] >= qty:
                    filled = True
                    break
                    
            if filled:
                print(f"[SUCCESS] {tk} 매수 완료!")
                current_holdings.append(tk)
                break
            else:
                retry_count += 1
                if retry_count > 5:
                    print(f"[FAIL] {tk} 5회 재시도 실패. 포기.")
                    trader.cancel_all_open_orders(tk)
                    break
                print(f"[WAIT] {tk} 매수 미체결. 취소 후 재시도 ({retry_count})")
                trader.cancel_all_open_orders(tk)
                time.sleep(1)
                
    usd_cash = trader.get_overseas_deposit()
    print(f"\n[남은 잔돈 영혼보내기] 현재 잔돈: ${usd_cash:,.2f}")
    if usd_cash > 100:
        for tk in candidates[:10]:
            if usd_cash < 100: break
            if tk in current_holdings:
                try: curr_px = yf.Ticker(tk).fast_info.last_price
                except: continue
                order_price = round(curr_px + 0.05, 2)
                extra_qty = int(usd_cash // order_price)
                if extra_qty > 0:
                    print(f"[추가매수] {tk} {extra_qty}주")
                    trader.send_order(tk, "BUY", extra_qty, order_price)
                    time.sleep(5)
                    usd_cash = trader.get_overseas_deposit()

if __name__ == "__main__":
    buy_missing()
