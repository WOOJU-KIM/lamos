import time
import sqlite3
import yfinance as yf
from datetime import datetime
from trader import KiwoomRESTTrader

def rebalance():
    trader = KiwoomRESTTrader()
    
    print("=== 1단계: 전량 매도 (기존 포트폴리오 청산) ===")
    print("이미 매도 완료됨. 2단계로 넘어갑니다.")

    print("\n=== 2단계: 신규 15분할 매수 ===")
    usd_cash = trader.get_overseas_deposit()
    print(f"확보된 총 예수금: ${usd_cash:,.2f}")
    
    conn = sqlite3.connect("quant_system.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ticker FROM daily_candidates WHERE date = date('now') ORDER BY rank ASC LIMIT 15")
    candidates = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    empty_slots = 15
    my_symbols = []
    buy_amount = usd_cash / empty_slots
    
    for tk in candidates:
        retry_count = 0
        while True:
            try:
                curr_px = yf.Ticker(tk).fast_info.last_price
            except:
                time.sleep(1)
                continue
                
            order_price = round(curr_px + 0.05, 2)
            qty = int(buy_amount // order_price)
            
            if qty <= 0:
                print(f"[PASS] {tk} 매수 금액 부족")
                break
                
            print(f"[매수] {tk} {qty}주 @ ${order_price:.2f} (현재가 +0.05)")
            trader.send_order(tk, "BUY", qty, order_price)
            
            filled = False
            for _ in range(10):
                time.sleep(1)
                pf = trader.get_portfolio()
                tk_holding = next((x for x in pf if x["symbol"] == tk), None)
                if tk_holding and tk_holding["qty"] >= qty:
                    filled = True
                    break
                    
            if filled:
                print(f"[SUCCESS] {tk} 매수 완료!")
                my_symbols.append(tk)
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
            if tk in my_symbols:
                try: curr_px = yf.Ticker(tk).fast_info.last_price
                except: continue
                order_price = round(curr_px + 0.05, 2)
                extra_qty = int(usd_cash // order_price)
                if extra_qty > 0:
                    print(f"[추가매수] {tk} {extra_qty}주")
                    trader.send_order(tk, "BUY", extra_qty, order_price)
                    time.sleep(5)
                    usd_cash = trader.get_overseas_deposit()
                    
    print("\n=== 모든 리밸런싱 작업이 완료되었습니다! ===")

if __name__ == "__main__":
    rebalance()
