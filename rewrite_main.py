import re

def rewrite_main():
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()

    new_code = """
                    # 1차 매수 (N빵)
                    for tk in candidates:
                        if empty_slots <= 0:
                            break
                        if tk not in my_symbols:
                            import yfinance as yf
                            
                            retry_count = 0
                            while True:
                                try:
                                    curr_px = yf.Ticker(tk).fast_info.last_price
                                except:
                                    break
                                    
                                order_price = round(curr_px + 0.05, 2)
                                qty = int(buy_amount // order_price)
                                if qty <= 0:
                                    break
                                    
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] 1차 신규 매수 주문: {tk} {qty}주 @ ${order_price:.2f} (현재가 +0.05)")
                                trader.send_order(tk, "BUY", qty, order_price)
                                
                                filled = False
                                print(f"[{tk}] 체결 및 잔고 반영 확인 중 (최대 10초)...")
                                for sec in range(10):
                                    time.sleep(1)
                                    pf = trader.get_portfolio()
                                    if any(p["symbol"] == tk for p in pf):
                                        filled = True
                                        break
                                        
                                if filled:
                                    print(f"✅ [{tk}] 매수 체결 완료!")
                                    my_symbols.append(tk)
                                    empty_slots -= 1
                                    usd_cash -= (qty * order_price)
                                    break
                                else:
                                    retry_count += 1
                                    if retry_count > 5:
                                        print(f"❌ [{tk}] 5회 재시도 실패. 패스합니다.")
                                        trader.cancel_all_open_orders(tk)
                                        break
                                    print(f"⏳ [{tk}] 10초 내 미체결. 취소 후 현재가 갱신하여 재시도 (시도 {retry_count})...")
                                    trader.cancel_all_open_orders(tk)
                                    time.sleep(1)

                    usd_cash = trader.get_overseas_deposit()
                    
                    # 2차 매수 (남은 짤짤이 현금 영혼 보내기)
                    if usd_cash > MIN_TRADE_USD:
                        for tk in candidates[:10]:
                            if usd_cash < MIN_TRADE_USD:
                                break
                                
                            if tk in my_symbols:
                                try:
                                    curr_px = yf.Ticker(tk).fast_info.last_price
                                except:
                                    continue
                                
                                retry_count = 0
                                while True:
                                    order_price = round(curr_px + 0.05, 2)
                                    extra_qty = int(usd_cash // order_price)
                                    if extra_qty <= 0:
                                        break
                                        
                                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 2차 영혼보내기 주문: {tk} {extra_qty}주 @ ${order_price:.2f}")
                                    
                                    prev_qty = 0
                                    for p in trader.get_portfolio():
                                        if p["symbol"] == tk:
                                            prev_qty = p["qty"]
                                            
                                    trader.send_order(tk, "BUY", extra_qty, order_price)
                                    
                                    filled = False
                                    print(f"[{tk}] 추가 체결 확인 중 (최대 10초)...")
                                    for sec in range(10):
                                        time.sleep(1)
                                        pf = trader.get_portfolio()
                                        for p in pf:
                                            if p["symbol"] == tk and p["qty"] > prev_qty:
                                                filled = True
                                                break
                                        if filled:
                                            break
                                            
                                    if filled:
                                        print(f"✅ [{tk}] 2차 영혼보내기 체결 완료!")
                                        usd_cash = trader.get_overseas_deposit()
                                        break
                                    else:
                                        retry_count += 1
                                        if retry_count > 5:
                                            print(f"❌ [{tk}] 2차 영혼보내기 5회 실패. 패스.")
                                            trader.cancel_all_open_orders(tk)
                                            break
                                        print(f"⏳ [{tk}] 10초 내 미체결. 취소 후 재시도 (시도 {retry_count})...")
                                        trader.cancel_all_open_orders(tk)
                                        try:
                                            curr_px = yf.Ticker(tk).fast_info.last_price
                                        except:
                                            pass
                                        time.sleep(1)
                                        usd_cash = trader.get_overseas_deposit()
"""

    old_block_pattern = re.compile(r"                    # 1차 매수 \(N빵\).*?(?=\s*except Exception as e:)", re.DOTALL)
    new_content = old_block_pattern.sub(new_code, content)
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Done replacing.')

if __name__ == "__main__":
    rewrite_main()
