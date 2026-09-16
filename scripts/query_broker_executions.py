import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kis_client import KisClient

print("=" * 80)
print("🏛 [한국투자증권(KIS) OpenAPI 실시간 체결내역 직접 통신 조회]")
print("=" * 80)

kis = KisClient(mode="VIRTUAL")

today_str = datetime.now().strftime("%Y%m%d")
start_str = (datetime.now() - timedelta(days=2)).strftime("%Y%m%d")

print(f"• 조회 대상 계좌: {kis.cano}-{kis.acnt_prdt_cd}")
print(f"• 조회 서버 URL: {kis.base_url}")
print(f"• 조회 기간: {start_str} ~ {today_str}")

res = kis.inquire_ccnl(start_date=start_str, end_date=today_str)

print(f"\n📡 [증권사 통신 결과]")
print(f"• API 통신 상태: {'✅ 성공 (HTTP 200 / rt_cd: 0)' if res.get('ok') else '❌ 실패'}")
print(f"• 체결/주문 건수: {res.get('count', 0)}건")

executions = res.get("executions", [])
if executions:
    print("\n📋 [증권사 공식 체결 내역 상세]")
    for idx, ex in enumerate(executions):
        sym = ex.get("pdno", "")
        name = ex.get("prdt_name", sym)
        side_cd = ex.get("sll_buy_dvsn_cd", "")
        side = "매수 (BUY)" if side_cd == "02" else ("매도 (SELL)" if side_cd == "01" else "기타")
        qty = ex.get("ft_ccld_qty") or ex.get("ccld_qty") or 0
        px = ex.get("ft_ccld_unpr3") or ex.get("ccld_unpr") or 0.0
        dt = ex.get("ord_dt", "")
        tm = ex.get("ord_tmd", "")
        odno = ex.get("odno", "")
        print(f" [{idx+1}] 주문번호: {odno} | 일시: {dt} {tm} | {name}({sym}) {side} | 체결: {qty}주 @ ${float(px):.2f}")
else:
    print("\n📋 [증권사 공식 체결 내역 상세]")
    print("• 금일 체결 내역: 0건 (체결 완료된 주문 없음 / 100% 현금 보존)")

# 실시간 예수금 잔고 조회
dep = kis.inquire_deposit()
bal = kis.inquire_balance()

avail_usd = float(dep.get("avail_usd", 0.0))
krw_conv = int(dep.get("krw_converted", int(avail_usd * 1402.5)))
holdings = bal.get("holdings", [])

print(f"\n💰 [실시간 계좌 잔고 현황]")
print(f"• 주문가능 외화: ${avail_usd:,.2f} USD")
print(f"• 원화 환산 잔고: {krw_conv:,}원")
print(f"• 보유 주식 수: {len(holdings)}개")

print("=" * 80)
