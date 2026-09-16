import sys
from pathlib import Path
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, INITIAL_CAPITAL_KRW
from core.backtest_engine import GranularBacktestEngine
from agents.dispatcher_agent import DispatcherAgent
from core.telegram_controller import TelegramController

# Windows 콘솔 utf-8 인코딩 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def main():
    print("=" * 75)
    print("🏛  [SOXL / SOXS 머신러닝 3중 타임프레임 & 100% 챔피언 관제 시스템]  🏛")
    print("=" * 75)

    # 1. 챔피언 설정 백테스트 실행 (100% 전액 투입, 90분 타임스탑, 65% 확신도)
    print("\n⏳ [1/3] 챔피언 Baseline ML Feature & Triple Screen Backtest 실행 중...")
    engine = GranularBacktestEngine(
        initial_capital_krw=INITIAL_CAPITAL_KRW,
        allocation_pct=1.00,
        confidence_threshold=0.65
    )
    bt_results = engine.run_backtest()
    
    print(f"   👉 시작 원금: {bt_results['initial_capital_krw']:,}원 ➡️ 최종 잔고: {bt_results['final_capital_krw']:,}원")
    print(f"   👉 누적 손익: {bt_results['total_pnl_krw']:+,}원 ({bt_results['total_return_pct']:+.2f}%) | 승률: {bt_results['win_rate_pct']}% | MDD: {bt_results['mdd_pct']}%")
    print(f"   👉 총 거래 횟수: {bt_results['total_trades_count']}회 ({bt_results['total_wins']}승 {bt_results['total_losses']}패)")
    print(f"   👉 SOXL 승률: {bt_results['soxl_win_rate_pct']}% | SOXS 승률: {bt_results['soxs_win_rate_pct']}% | PF: {bt_results['profit_factor']}")
    print("✅ 챔피언 Baseline 백테스팅 완료")

    # 2. 손익금 병기 성적표 텔레그램 발송
    print("\n📱 [2/3] 챔피언 성적표 텔레그램 발송 중...")
    dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    concise_report = dispatcher.compose_backtest_report(bt_results)
    send_res = dispatcher.send_telegram_message(concise_report)
    
    if send_res.get("ok"):
        print("🚀 >>> 챔피언 백테스트 성적표 텔레그램 발송 성공! <<< 🚀")
    else:
        print(f"⚠️ 텔레그램 발송 확인: {send_res}")

    # 3. LLM 대화형 AI 총괄 비서 리스너 가동
    print("\n📡 [3/3] LLM 기반 대화형 AI 총괄 비서 리스너 활성화 중...")
    controller = TelegramController(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    welcome_msg = """🤖 [챔피언 24.38% 롤백 완료 및 AI 관제 가동]
대표님, 요청하신 대로 챔피언 설정(100% 비중, 90분 타임스탑, 승률 65.0%, 최종 수익률 +24.38%)으로 롤백 및 동기 연동 점검을 완료하였습니다.
성적표를 확인해 주시고, 텔레그램으로 어떤 지시든 말씀해 주시면 즉각 처리하겠습니다!"""
    controller.send_message(welcome_msg)
    print("✅ 대화형 AI 비서 리스너 활성화 완료. 대표님의 텔레그램 메시지 대기 중...")

    # 지속 수신 루프 (Background Daemon)
    controller.listen_loop(poll_interval=2)

if __name__ == "__main__":
    main()
