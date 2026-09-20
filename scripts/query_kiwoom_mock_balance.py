import sys
import json
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker

print("=" * 80)
print("🏛 [키움증권(Kiwoom) 모의투자 계좌 실시간 잔고 & 예수금 직접 조회]")
print("=" * 80)

kb = KiwoomBroker(is_simulation=True)

print(f"• 서버 URL: {kb.base_url}")
print(f"• 모의 계좌번호: {kb.account_no}-{kb.account_type}")

# 1. OAuth2 토큰 발급
token = kb.get_access_token(force_refresh=True)
print(f"• OAuth2 토큰 발급: ✅ 성공 (Prefix: {token[:12]}...)")

# 2. 국내주식 외화 예수금 조회 (kt00001)
dep = kb.get_domestic_deposit()
print("\n💵 [1. 국내주식 외화 예수금 조회 (TR: kt00001)]")
print(f"• API 통신 결과: {'✅ 성공' if dep.get('ok') else '❌ 실패'}")
print(f"• USD 외화 예수금: ${dep.get('usd_deposit', 0.0):,.2f} USD")
print(f"• USD 주문가능금액: ${dep.get('usd_order_available', 0.0):,.2f} USD")
print(f"• 원화 환산 잔고: {dep.get('krw_converted', 0):,}원")
print(f"• 응답 메시지: {dep.get('msg')}")

raw_dep = dep.get("raw_response", {})
print(f"• 증권사 반환 원본 통화 목록: {json.dumps(dep.get('all_currencies', []), ensure_ascii=False, indent=2)}")

# 3. 국내주식 원장 잔고 조회 (kt00018)
bal = kb.get_domestic_stock_balance()
print("\n📈 [2. 국내주식 원장 보유 잔고 조회 (TR: kt00018)]")
print(f"• API 통신 결과: {'✅ 성공' if bal.get('ok') else '❌ 실패'}")
print(f"• 총 주식 평가금액: ${bal.get('total_eval_usd', 0.0):,.2f} USD")
print(f"• 총 매입 금액: ${bal.get('total_purchase_usd', 0.0):,.2f} USD")
print(f"• 보유 종목 수: {bal.get('holdings_count', 0)}개")
print(f"• 보유 종목 목록: {json.dumps(bal.get('holdings', []), ensure_ascii=False)}")
print(f"• 응답 메시지: {bal.get('msg')}")

# 4. 국내 예수금 조회 (kt00001)
try:
    kr_dep = kb.get_domestic_deposit()
    print("\n🇰🇷 [3. 국내 예수금 조회 (TR: kt00001)]")
    print(f"• 예수금(D+2): {kr_dep.get('deposit_d2', 0):,}원 | 주문가능금액: {kr_dep.get('order_available_krw', 0):,}원")
except Exception as e:
    print(f"\n🇰🇷 [3. 국내 예수금 조회]: {e}")

print("=" * 80)
