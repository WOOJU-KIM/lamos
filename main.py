import time
from datetime import datetime
import pytz
import sqlite3
from scanner import run_premarket_scan
from trader import KiwoomRESTTrader
from config import DB_PATH, MAX_HOLDINGS, TAKE_PROFIT_PCT, STOP_LOSS_PCT, MIN_TRADE_USD, TRADE_LOOP_INTERVAL_SEC

def is_dst():
    """미국 동부 시간 기준 서머타임 여부 반환"""
    tz = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    return now.dst() != type(now.dst())(0)

def get_market_hours():
    """현재 날짜 기준 장 시작/종료 시간 반환 (KST 기준)"""
    dst = is_dst()
    # 서머타임: 22:30 ~ 05:00 / 해제시: 23:30 ~ 06:00
    open_hour, open_minute = (22, 30) if dst else (23, 30)
    close_hour, close_minute = (5, 0) if dst else (6, 0)
    return open_hour, open_minute, close_hour, close_minute

def is_market_open(now, open_h, open_m, close_h, close_m):
    """현재 시간이 장 중인지 확인"""
    current_minutes = now.hour * 60 + now.minute
    open_minutes = open_h * 60 + open_m
    close_minutes = close_h * 60 + close_m
    
    # 자정을 넘기는 시간대이므로 계산 처리
    if current_minutes >= open_minutes or current_minutes < close_minutes:
        return True
    return False

def run_system():
    print("==================================================")
    print("Lamos Quant System Started (Auto DST Support)")
    print("==================================================")
    
    trader = None
    scanned_today = False
    
    while True:
        now = datetime.now()
        open_h, open_m, close_h, close_m = get_market_hours()
        
        # 1. 장 시작 30분 전: 프리마켓 스캔 (종목 발굴)
        scan_h = open_h if open_m >= 30 else (open_h - 1)
        scan_m = (open_m - 30) if open_m >= 30 else (open_m + 30)
        
        if now.hour == scan_h and now.minute == scan_m and not scanned_today:
            print(f"\n[{now.strftime('%H:%M:%S')}] 장 시작 30분 전! Z-Score 스캔 시작...")
            try:
                run_premarket_scan()
                scanned_today = True
            except Exception as e:
                print(f"스캔 중 에러 발생: {e}")
                
        # 2. 장 마감 후 스캔 플래그 초기화
        if now.hour == close_h and now.minute == close_m:
            scanned_today = False
            if trader is not None:
                print(f"\n[{now.strftime('%H:%M:%S')}] 정규장 마감. 트레이더 휴식 모드 진입.")
                trader = None # 인스턴스 초기화 (다음 날 새 토큰 발급)
                time.sleep(TRADE_LOOP_INTERVAL_SEC) # 중복 실행 방지
                continue

        # 3. 장 중: 트레이딩 로직 실행
        if is_market_open(now, open_h, open_m, close_h, close_m):
            if trader is None:
                print(f"\n[{now.strftime('%H:%M:%S')}] 미국 본장 오픈! 트레이더 인스턴스 가동 (토큰 발급)...")
                try:
                    trader = KiwoomRESTTrader()
                except Exception as e:
                    print(f"트레이더 초기화 실패: {e}")
                    time.sleep(TRADE_LOOP_INTERVAL_SEC)
                    continue

            try:
                holdings = trader.get_portfolio()
                usd_cash = trader.get_overseas_deposit()
                
                # [매도 체크]
                for pos in holdings:
                    ret = (pos["now_price"] - pos["avg_price"]) / pos["avg_price"] if pos["avg_price"] > 0 else 0
                    if ret >= TAKE_PROFIT_PCT or ret <= STOP_LOSS_PCT:
                        print(f"[{pos['symbol']}] 목표 조건 도달 (수익률 {ret*100:.2f}%). 매도 진행.")
                        sell_price = round(pos["now_price"] - 0.03, 2)
                        trader.send_order(pos["symbol"], "SELL", pos["qty"], sell_price)
                        
                # [신규 매수 체크] 목표 종목 수 유지
                empty_slots = MAX_HOLDINGS - len(holdings)
                if empty_slots > 0 and usd_cash > MIN_TRADE_USD:
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT ticker FROM daily_candidates WHERE date = date('now') ORDER BY rank ASC")
                    candidates = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    
                    my_symbols = [p["symbol"] for p in holdings]
                    buy_amount = usd_cash / empty_slots
                    

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
                                    print(f"[SUCCESS] [{tk}] 매수 체결 완료!")
                                    my_symbols.append(tk)
                                    empty_slots -= 1
                                    usd_cash -= (qty * order_price)
                                    break
                                else:
                                    retry_count += 1
                                    if retry_count > 5:
                                        print(f"[FAIL] [{tk}] 5회 재시도 실패. 패스합니다.")
                                        trader.cancel_all_open_orders(tk)
                                        break
                                    print(f"[WAIT] [{tk}] 10초 내 미체결. 취소 후 현재가 갱신하여 재시도 (시도 {retry_count})...")
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
                                        print(f"[SUCCESS] [{tk}] 2차 영혼보내기 체결 완료!")
                                        usd_cash = trader.get_overseas_deposit()
                                        break
                                    else:
                                        retry_count += 1
                                        if retry_count > 5:
                                            print(f"[FAIL] [{tk}] 2차 영혼보내기 5회 실패. 패스.")
                                            trader.cancel_all_open_orders(tk)
                                            break
                                        print(f"[WAIT] [{tk}] 10초 내 미체결. 취소 후 재시도 (시도 {retry_count})...")
                                        trader.cancel_all_open_orders(tk)
                                        try:
                                            curr_px = yf.Ticker(tk).fast_info.last_price
                                        except:
                                            pass
                                        time.sleep(1)
                                        usd_cash = trader.get_overseas_deposit()

                                    
            except Exception as e:
                print(f"장중 트레이딩 로직 에러: {e}")
                
        time.sleep(TRADE_LOOP_INTERVAL_SEC)

if __name__ == "__main__":
    run_system()
