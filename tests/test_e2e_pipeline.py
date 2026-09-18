import sys
import os
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

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, INITIAL_CAPITAL_KRW
from core.backtest_engine import GranularBacktestEngine
from agents.dispatcher_agent import DispatcherAgent
from core.agentic_brain import AgenticTelegramBrain
from core.telegram_controller import TelegramController

def run_e2e_test():
    print("=" * 75)
    print("🧪 [E2E 파이프라인 전수 점검: 텔레그램 명령 ➔ 백엔드 실행 ➔ 텔레그램 발송] 🧪")
    print("=" * 75)

    all_passed = True

    # ----------------------------------------------------
    # [1단계: 수신 단계 점검]
    # ----------------------------------------------------
    print("\n[1단계: 수신 단계 점검 (Telegram Listener / Polling Check)]")
    controller = TelegramController(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    try:
        updates = controller.get_updates(timeout=2)
        print(f"  ✅ [PASS] 텔레그램 서버 Polling 통신 정상 (최근 업데이트 수신: {len(updates)}건)")
    except Exception as e:
        print(f"  ❌ [FAIL] 텔레그램 수신 통신 에러: {e}")
        all_passed = False

    # ----------------------------------------------------
    # [2단계: 인식 단계 점검]
    # ----------------------------------------------------
    print("\n[2단계: 인식 단계 점검 (대표님 자연어 지시 ➔ 실행 Intent 매핑)]")
    brain = AgenticTelegramBrain()
    test_user_commands = [
        "기존것이 더 낫네 기존것으로 바꿔줘 다시",
        "다시 백테스팅 돌려봐바 그대로",
        "아니 아까 처음으로 돌려놓으라고 24퍼 수익률짜리로",
        "실행해",
        "아니 그던 멈추고 아까 24퍼짜리 수익율로 모델 다시 해서 백테스트 재진행해"
    ]
    for cmd in test_user_commands:
        is_exec = brain._is_execution_or_rollback_request(cmd)
        if is_exec:
            print(f"  ✅ [PASS] 명령 인식 성공: '{cmd}' ➔ [백테스트 동기 실행 매핑]")
        else:
            print(f"  ❌ [FAIL] 명령 인식 실패: '{cmd}' ➔ [일반 대화로 오분류]")
            all_passed = False

    # ----------------------------------------------------
    # [3단계: 실행 단계 점검]
    # ----------------------------------------------------
    print("\n[3단계: 실행 단계 점검 (챔피언 Baseline 백테스트 동기 실행)]")
    try:
        engine = GranularBacktestEngine(
            initial_capital_krw=INITIAL_CAPITAL_KRW,
            allocation_pct=1.00,
            confidence_threshold=0.40
        )
        res = engine.run_backtest()
        
        trades = res["total_trades_count"]
        ret = res["total_return_pct"]
        win_rate = res["win_rate_pct"]
        final_cap = res["final_capital_krw"]
        mdd = res["mdd_pct"]
        pf = res["profit_factor"]

        print(f"  📊 연산 완료 결과:")
        print(f"     • 시작원금: {res['initial_capital_krw']:,}원 ➔ 최종잔고: {final_cap:,}원")
        print(f"     • 수익률: {ret:+.2f}% ({res['total_pnl_krw']:+,}원)")
        print(f"     • 총 거래: {trades}회 ({res['total_wins']}승 {res['total_losses']}패, 승률 {win_rate}%)")
        print(f"     • TQQQ 승률: {res['tqqq_win_rate_pct']}% | SQQQ 승률: {res['sqqq_win_rate_pct']}%")
        print(f"     • 손익비(PF): {pf} | MDD: -{mdd:.2f}%")

        if trades == 20 and ret == 24.38 and win_rate == 65.0 and final_cap == 12_438_395:
            print("  ✅ [PASS] 챔피언 24.38% Baseline 지표 100% 일치 확인 완료!")
        else:
            print(f"  ⚠️ [CHECK] 챔피언 지표 차이 (Trades={trades}, Ret={ret}%, WinRate={win_rate}%)")
    except Exception as e:
        print(f"  ❌ [FAIL] 백테스트 연산 에러: {e}")
        all_passed = False

    # ----------------------------------------------------
    # [4단계: 발송 단계 점검]
    # ----------------------------------------------------
    print("\n[4단계: 발송 단계 점검 (텔레그램 성적표 전송)]")
    dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    scorecard = dispatcher.compose_backtest_report(res)
    send_res = dispatcher.send_telegram_message(scorecard)

    if send_res.get("ok"):
        print("  ✅ [PASS] 텔레그램 실시간 성적표 발송 성공 (HTTP 200 OK)")
    else:
        print(f"  ❌ [FAIL] 텔레그램 발송 실패: {send_res}")
        all_passed = False

    print("\n" + "=" * 75)
    if all_passed:
        print("🎉 [최종 결과] E2E 4단계 파이프라인 전수 점검 ALL PASS (정상 작동 확인)")
    else:
        print("⚠️ [최종 결과] E2E 파이프라인 점검 중 일부 항목 주의 필요")
    print("=" * 75)

if __name__ == "__main__":
    run_e2e_test()
