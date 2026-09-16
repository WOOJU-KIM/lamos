import yfinance as yf
import pandas as pd
from typing import Dict, Any
from datetime import datetime

class DataAgent:
    """
    [1. Data Agent]
    yfinance 라이브러리를 통해 실시간 및 1~3년 장기 SOXL, SOXS, SOXX, VIX 데이터를 수집하고
    정량 지표를 종합 정제하는 에이전트
    """
    def __init__(self):
        self.symbols = {
            "SOXL": "SOXL",
            "SOXS": "SOXS",
            "SOXX": "SOXX",
            "VIX": "^VIX"
        }

    def fetch_market_data(self) -> Dict[str, Any]:
        """실시간 및 장단기 시계열 데이터 수집"""
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "assets": {},
            "vix": {},
            "raw_dfs": {}
        }

        for name, ticker_sym in self.symbols.items():
            ticker = yf.Ticker(ticker_sym)
            hist = ticker.history(period="1y")
            if hist.empty:
                # 백업 기간 시도
                hist = ticker.history(period="6mo")
                
            results["raw_dfs"][name] = hist
            
            latest_row = hist.iloc[-1]
            prev_row = hist.iloc[-2] if len(hist) > 1 else latest_row
            
            current_price = round(float(latest_row['Close']), 2)
            prev_close = round(float(prev_row['Close']), 2)
            change_amount = round(current_price - prev_close, 2)
            change_pct = round((change_amount / prev_close) * 100, 2) if prev_close != 0 else 0.0
            
            day_high = round(float(latest_row['High']), 2)
            day_low = round(float(latest_row['Low']), 2)
            volume = int(latest_row['Volume'])
            
            ma5 = round(float(hist['Close'].tail(5).mean()), 2) if len(hist) >= 5 else current_price
            ma20 = round(float(hist['Close'].tail(20).mean()), 2) if len(hist) >= 20 else current_price
            ma50 = round(float(hist['Close'].tail(50).mean()), 2) if len(hist) >= 50 else current_price
            ma200 = round(float(hist['Close'].tail(200).mean()), 2) if len(hist) >= 200 else current_price
            
            data_dict = {
                "symbol": ticker_sym,
                "current_price": current_price,
                "prev_close": prev_close,
                "change_amount": change_amount,
                "change_pct": change_pct,
                "day_high": day_high,
                "day_low": day_low,
                "volume": volume,
                "ma5": ma5,
                "ma20": ma20,
                "ma50": ma50,
                "ma200": ma200
            }
            
            if name == "VIX":
                results["vix"] = data_dict
            else:
                results["assets"][name] = data_dict

        return results

    def get_summary_text(self, market_data: Dict[str, Any]) -> str:
        """에이전트에 주입할 텍스트 요약본 생성"""
        soxl = market_data["assets"]["SOXL"]
        soxs = market_data["assets"]["SOXS"]
        vix = market_data["vix"]

        summary = f"""[실시간 시장 데이터 브리핑 - {market_data['timestamp']}]
1. SOXL (반도체 3x 불):
   - 현재가: ${soxl['current_price']} (전일 대비 {soxl['change_pct']:+.2f}%, {soxl['change_amount']:+.2f}$)
   - 당일 범위: 저가 ${soxl['day_low']} ~ 고가 ${soxl['day_high']}
   - 기술적 지표: 5일선 ${soxl['ma5']} | 20일선 ${soxl['ma20']} | 50일선 ${soxl['ma50']} | 200일선 ${soxl['ma200']}

2. SOXS (반도체 3x 베어/헤지):
   - 현재가: ${soxs['current_price']} (전일 대비 {soxs['change_pct']:+.2f}%, {soxs['change_amount']:+.2f}$)
   - 당일 범위: 저가 ${soxs['day_low']} ~ 고가 ${soxs['day_high']}
   - 기술적 지표: 5일선 ${soxs['ma5']} | 20일선 ${soxs['ma20']}

3. VIX (시장 변동성 지수):
   - 현재 지수: {vix['current_price']} pt (전일 대비 {vix['change_pct']:+.2f}%)
   - 당일 범위: {vix['day_low']} ~ {vix['day_high']}
"""
        return summary
