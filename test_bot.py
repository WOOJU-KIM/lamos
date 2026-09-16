import os
import requests
import yfinance as yf
from dotenv import load_dotenv

# .env 파일에서 키 불러오기
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message: str):
    """텔레그램으로 메시지를 발송하는 함수"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    response = requests.post(url, json=payload)
    return response.json()

# 1. TQQQ 및 SQQQ 실시간 데이터 조회
tqqq = yf.Ticker("TQQQ").history(period="1d")
sqqq = yf.Ticker("SQQQ").history(period="1d")

tqqq_price = round(tqqq['Close'].iloc[-1], 2)
sqqq_price = round(sqqq['Close'].iloc[-1], 2)

# 2. 대표님께 보낼 브리핑 메시지 생성
report_text = f"""🏛 *[AI 퀀트 비서실 시스템 가동]*

대표님, AI 퀀트 시스템과 텔레그램 관제탑 연결이 성공적으로 완료되었습니다!

📊 *실시간 주요 종목 모니터링*
• **TQQQ 현재가:** `${tqqq_price}`
• **SQQQ 현재가:** `${sqqq_price}`

대표님의 지침(TQQQ/SQQQ 선제 헤지 및 스위칭 전략)을 수행할 준비가 되었습니다.
지금부터 백테스팅 및 실시간 감시 파이프라인을 가동합니다. 🚀
"""

# 3. 텔레그램으로 전송
res = send_telegram_message(report_text)
if res.get("ok"):
    print(">>> 텔레그램 발송 성공! 스마트폰을 확인해 보세요.")
else:
    print(f">>> 발송 실패: {res}")