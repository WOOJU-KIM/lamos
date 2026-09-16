import sys
from datetime import datetime
from pathlib import Path

# UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.system_logger import system_logger
from core.live_runner import USMarketCalendar
from core.hybrid_broker import HybridUniversalBroker

print("=" * 70)
print("🏛 [Lumos 모의 거래 실시간 가동 현황 점검]")
print("=" * 70)

mkt = USMarketCalendar.get_market_status()
print(f"• 현재 시각: {mkt['now_kst_str']}")
print(f"• 세션 상태: {mkt['status_desc']}")
print(f"• 개장 대기: {mkt['time_until_open_str']}")

broker = HybridUniversalBroker(is_simulation=True)
conn = broker.test_connection()
print(f"• 브로커 연동: {conn['broker_name']} ({conn['mode']})")
print(f"• 주문가능 예수금: ${conn['usd_order_available']:,.2f} USD (₩{conn['krw_converted']:,}원)")
print(f"• 보유 종목 수: {conn['holdings_count']}개 (100% 현금 대기)")

print("\n📜 [최근 시스템 로그 8건]")
for l in system_logger.get_logs(limit=8):
    print(f"  [{l['timestamp']}][{l['level']}][{l['source']}] {l['message']}")
print("=" * 70)
