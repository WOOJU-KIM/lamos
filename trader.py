import os
import requests
import json
import sqlite3
import time
from datetime import datetime
from database import DB_PATH, log_trade

# ------------------------------------------------------------------------------
# 환경 변수 불러오기 (.env)
# ------------------------------------------------------------------------------
def load_env():
    env_vars = {}
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                env_vars[key] = val
    return env_vars

env = load_env()

APP_KEY = env.get('KIWOOM_MOCK_APP_KEY')
APP_SECRET = env.get('KIWOOM_MOCK_APP_SECRET')
CANO = env.get('KIWOOM_MOCK_ACCOUNT_NO')
ACNT_TYPE = env.get('KIWOOM_ACCOUNT_TYPE', '01')
BASE_URL = "https://mockapi.kiwoom.com"

# ------------------------------------------------------------------------------
# Kiwoom REST API 클래스 (Lumos 참조 완벽 구현)
# ------------------------------------------------------------------------------
class KiwoomRESTTrader:
    def __init__(self):
        self.access_token = self.get_access_token()
        
    def get_access_token(self):
        """키움 REST API 토큰 발급"""
        url = f"{BASE_URL}/oauth2/token"
        headers = {"Content-Type": "application/json;charset=UTF-8"}
        body = {
            "grant_type": "client_credentials",
            "appkey": APP_KEY,
            "secretkey": APP_SECRET  # KIS의 appsecret과 다르게 secretkey를 사용
        }
        
        print("키움증권 REST API 토큰 발급 요청 중...")
        res = requests.post(url, headers=headers, json=body)
        if res.status_code == 200:
            data = res.json()
            token = data.get('token') or data.get('access_token')
            print("O Kiwoom Token Success!")
            return token
        else:
            raise RuntimeError(f"토큰 발급 실패: {res.text}")

    def _send_tr(self, endpoint, api_id, body_dict):
        url = f"{BASE_URL}{endpoint}"
        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "api-id": api_id,
            "cont-yn": "N",
            "next-key": ""
        }
        try:
            res = requests.post(url, headers=headers, json=body_dict, timeout=10)
            return res.json()
        except Exception as e:
            print(f"API 요청 중 에러 또는 타임아웃: {e}")
            return {}

    def get_overseas_deposit(self):
        """[ust21110] 해외주식 예수금 조회"""
        body = {"cano": str(CANO), "acnt_prdt_cd": str(ACNT_TYPE)}
        res = self._send_tr("/api/us/acnt", "ust21110", body)
        
        usd_deposit = 0.0
        for item in res.get("result_list", []):
            if item.get("crnc_code") == "USD":
                usd_deposit = float(str(item.get("fc_ord_alowa", "0")).replace(",", ""))
                break
        return usd_deposit

    def get_portfolio(self):
        """[ust21070] 해외주식 잔고 조회"""
        body = {"cano": str(CANO), "acnt_prdt_cd": str(ACNT_TYPE), "qry_tp": "1"}
        res = self._send_tr("/api/us/acnt", "ust21070", body)
        
        holdings = []
        for item in res.get("result_list", []):
            qty_str = item.get("poss_qty")
            if not qty_str or int(qty_str) == 0:
                qty_str = item.get("qty") or "0"
            qty = int(float(str(qty_str).replace(",", "")))
            if qty > 0:
                holdings.append({
                    "symbol": str(item.get("stk_cd") or item.get("symbol") or "").strip().upper(),
                    "qty": qty,
                    "avg_price": float(str(item.get("purchase_price") or item.get("frgn_stk_book_uv") or "0").replace(",", "")),
                    "now_price": float(str(item.get("eval_price") or item.get("now_pric") or "0").replace(",", ""))
                })
        return holdings

    def send_order(self, symbol, order_type, qty, price, exchange=None):
        """[ust20000/ust20001] 주문 전송"""
        sym_clean = symbol.upper().strip()
        stex = exchange or ("NY" if sym_clean in ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "VICI", "LEN", "ROL", "MCD", "CCL", "PCG", "ROP", "EFX", "INTU", "CLX", "ZTS", "OTIS", "GIS", "HD"] else "ND")
        
        is_buy = (order_type.upper() == "BUY")
        api_id = "ust20000" if is_buy else "ust20001"
        
        body = {
            "cano": str(CANO),
            "acnt_prdt_cd": str(ACNT_TYPE),
            "stex_tp": stex,
            "stk_cd": sym_clean,
            "ord_qty": str(qty),
            "ord_uv": str(round(price, 2)),
            "trde_tp": "00" # 지정가 (모의투자는 지정가 필수)
        }
        res = self._send_tr("/api/us/ordr", api_id, body)
        
        # 거래소 매칭 에러(1903) 발생 시 반대 거래소(NY <-> ND)로 교차 재시도
        if res and res.get("return_code") == 7 and "1903" in res.get("return_msg", ""):
            print(f"[{sym_clean}] 거래소 매칭 실패. 반대 거래소로 교차 주문 시도...")
            alt_stex = "ND" if stex == "NY" else "NY"
            body["stex_tp"] = alt_stex
            res = self._send_tr("/api/us/ordr", api_id, body)
            
        return res
    def get_open_orders(self):
        """[ust21050] 미국주식 미체결 내역 조회"""
        body = {
            "cano": str(CANO),
            "acnt_prdt_cd": "01",
            "qry_tp": "0"
        }
        res = self._send_tr("/api/us/acnt", "ust21050", body)
        if res.get("return_code") == 0:
            return res.get("result_list", [])
        return []

    def cancel_order(self, order_no, symbol, exchange=None):
        """[ust20003] 미체결 주문 취소"""
        sym_clean = symbol.upper().strip()
        stex = exchange or ("NY" if sym_clean in ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "VICI", "LEN", "ROL", "MCD", "CCL", "PCG", "ROP", "EFX", "INTU", "CLX", "ZTS", "OTIS", "GIS", "HD"] else "ND")
        body = {
            "cano": str(CANO),
            "acnt_prdt_cd": "01",
            "orig_ord_no": order_no,
            "stex_tp": stex,
            "stk_cd": sym_clean,
            "ord_qty": "0",
            "ord_uv": "0",
            "trde_tp": "00"
        }
        res = self._send_tr("/api/us/ordr", "ust20003", body)
        if res and res.get("return_code") == 7 and "1903" in res.get("return_msg", ""):
            alt_stex = "ND" if stex == "NY" else "NY"
            body["stex_tp"] = alt_stex
            res = self._send_tr("/api/us/ordr", "ust20003", body)
        return res

    def cancel_all_open_orders(self, symbol=None):
        """모든 미체결 주문 (또는 특정 종목 미체결) 전량 취소"""
        orders = self.get_open_orders()
        for ord_info in orders:
            ord_sym = str(ord_info.get("stk_cd") or "").strip().upper()
            ord_no = str(ord_info.get("ord_no") or "").strip()
            stex_nm = str(ord_info.get("stex_nm") or "").strip()
            
            if symbol and symbol.upper() != ord_sym:
                continue
                
            if "뉴욕" in stex_nm or "아멕스" in stex_nm:
                stex_tp = "NY"
            elif "나스닥" in stex_nm:
                stex_tp = "ND"
            else:
                stex_tp = "NY" if ord_sym in ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "VICI", "LEN", "ROL", "MCD", "CCL", "PCG", "ROP", "EFX", "INTU", "CLX", "ZTS", "OTIS", "GIS", "HD"] else "ND"
                
            if ord_no:
                print(f"[CANCEL] [{ord_sym}] 미체결 주문({ord_no}) 취소 전송")
                self.cancel_order(ord_no, ord_sym, stex_tp)
                import time
                time.sleep(0.5)

# ------------------------------------------------------------------------------
# 메인 트레이딩 봇
# ------------------------------------------------------------------------------
def run_trading():
    print(f"[{CANO}] 키움증권 REST API (모의투자) 트레이딩 봇 가동 시작...")
    trader = KiwoomRESTTrader()
    
    print("실시간 가격 모니터링 및 주문 대기 중...")
    
    while True:
        try:
            holdings = trader.get_portfolio()
            usd_cash = trader.get_overseas_deposit()
            
            # 장중 매도 조건 체크 (+25%, -15%)
            for pos in holdings:
                ret = (pos["now_price"] - pos["avg_price"]) / pos["avg_price"]
                if ret >= 0.25 or ret <= -0.15:
                    print(f"[{pos['symbol']}] 목표 조건 도달 (수익률 {ret*100:.2f}%). 매도 진행.")
                    trader.send_order(pos["symbol"], "SELL", pos["qty"], pos["now_price"])
                    
            # 10종목 미만일 경우 매수 진행
            empty_slots = 10 - len(holdings)
            if empty_slots > 0 and usd_cash > 10:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT ticker FROM daily_candidates WHERE date = date('now') ORDER BY rank ASC")
                candidates = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                my_symbols = [p["symbol"] for p in holdings]
                buy_amount = usd_cash / empty_slots
                
                for tk in candidates:
                    if empty_slots <= 0:
                        break
                    if tk not in my_symbols:
                        # 임시로 시세를 위해 yfinance 사용 (키움 실시간 조회를 붙일 수 있음)
                        import yfinance as yf
                        curr_px = yf.Ticker(tk).fast_info.last_price
                        qty = int(buy_amount // curr_px)
                        
                        if qty > 0:
                            print(f"[{tk}] 신규 편입 진행 (1순위). {qty}주 매수 주문.")
                            trader.send_order(tk, "BUY", qty, curr_px)
                            empty_slots -= 1
                            
        except Exception as e:
            print(f"에러 발생: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    run_trading()
