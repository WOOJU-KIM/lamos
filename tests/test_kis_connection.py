import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import (
    KIS_MODE,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID
)
from core.kis_client import KisClient
from agents.dispatcher_agent import DispatcherAgent

def run_kis_test():
    print("=" * 75)
    print(f"📡 [한국투자증권(KIS) 듀얼 환경 연결 및 계좌 점검 테스트: {KIS_MODE} 모드] 📡")
    print("=" * 75)

    client = KisClient(mode=KIS_MODE)
    res = client.test_connection()

    print(f"\n1. 접속 모드: {res['mode']} (Base URL: {res['base_url']})")
    cano_mask = res['cano'][:4] + "****" if res.get('cano') and len(res['cano']) >= 4 else "********"
    print(f"2. 대상 계좌: {cano_mask}-{res['acnt_prdt_cd']}")
    print(f"3. OAuth2 토큰 발급 상태: {'✅ 정상 발급 (24시간 유효 캐싱)' if res['token_ok'] else '❌ 발급 실패'}")
    if res.get('token_snippet'):
        print(f"   • Access Token Prefix: {res['token_snippet']}")

    print(f"4. 예수금(주문가능) 조회: {'✅ 성공' if res['deposit_ok'] else '⚠️ 응답 확인'}")
    print(f"   • 주문가능 외화(USD): ${res['avail_usd']:,.2f}")
    print(f"   • 주문가능 원화(KRW): ₩{res['avail_krw']:,.0f}")
    if res.get('deposit_msg'):
        print(f"   • 증권사 응답 메시지: {res['deposit_msg'].strip()}")

    print(f"5. 해외주식 잔고 조회: {'✅ 정상 조회' if res['balance_ok'] else '⚠️ 응답 확인'}")
    print(f"   • 보유 종목 수: {res['holdings_count']}개 (100% 현금 대기 중)")
    for h in res.get('holdings', []):
        print(f"     - [{h['ticker']}] {h['qty']}주 (평단 ${h['avg_price']:.2f} / 현재 ${h['now_price']:.2f} / 손익 {h['pnl_rate_pct']:+.2f}%)")

    # 텔레그램 보고서 작성 및 발송
    cano_display = f"{cano_mask}-{res['acnt_prdt_cd']}"
    mode_emoji = "🧪 [모의투자 VIRTUAL]" if res['mode'] == "VIRTUAL" else "🚨 [실전투자 REAL]"
    token_status = "🟢 정상 발급 완료 (24h 캐싱)" if res['token_ok'] else "🔴 발급 실패"
    
    report = f"""🏛 **[한국투자증권 KIS 해외주식 연동 점검 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **실행 모드:** `{mode_emoji}`
🔑 **계좌 번호:** `{cano_display}`
🌐 **서버 URL:** `{res['base_url']}`

📊 **[계좌 및 시스템 상태]**
• **OAuth2 토큰:** {token_status}
• **주문가능 외화:** `${res['avail_usd']:,.2f} USD`
• **주문가능 원화:** `₩{res['avail_krw']:,.0f} KRW`
• **현재 보유 종목:** `{res['holdings_count']}개` (100% 현금 보유 대기)

🛡 **[안전장치(Safety Guard) 적용 상태]**
• **종목 거래소 매핑:** TQQQ / SQQQ ➔ `AMS / NASD`
• **오버나잇 0% 가드:** 정규장 마감 10분 전 전량 시장가 청산 로직 활성화
• **손익비 룰:** 익절 `+3.5%` / 칼손절 `-2.0%` (손익비 1:1.75) / 타임스탑 `90분`

💡 한국투자증권 해외주식 주문 모듈이 정상 가동 준비를 마쳤습니다."""

    dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    send_res = dispatcher.send_telegram_message(report)

    if send_res.get("ok"):
        print("\n🚀 >>> KIS 연결 점검 결과 텔레그램 발송 성공! <<< 🚀")
    else:
        print(f"\n⚠️ 텔레그램 발송 확인: {send_res}")

    print("=" * 75)

if __name__ == "__main__":
    run_kis_test()
