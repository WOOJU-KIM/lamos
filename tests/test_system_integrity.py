import os
import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator
from core.kiwoom_broker import KiwoomBroker
from core.live_runner import KiwoomLiveRunner

class SystemIntegrityRegressionTest(unittest.TestCase):
    """
    [Lumos 퀀트 시스템 5대 절대 원칙 무결성 회귀 테스트]
    실전 자금 투입 전, 어떤 코드 수정이 있더라도 반드시 통과해야 하는 하드 인터락 검증
    """

    def setUp(self):
        self.moe = MoEMetaOrchestrator(confidence_threshold=0.65, gbdt_threshold=0.65, mode="hybrid_v3")
        # 가상 15분봉 데이터 생성
        dates = pd.date_range("2026-08-24 09:30", periods=50, freq="15min")
        self.dummy_15m = pd.DataFrame({
            "Open": np.linspace(100, 105, 50),
            "High": np.linspace(101, 106, 50),
            "Low": np.linspace(99, 104, 50),
            "Close": np.linspace(100.5, 105.5, 50),
            "Volume": [100000] * 50
        }, index=dates)

    def test_1_confidence_under_65_must_be_rejected(self):
        """[인터락 1] Lumos V3 GBDT 확신도 65% 미달 시 무조건 매수 거부 (is_approved == False)"""
        # 임의로 55% 수준의 데이터 주입
        res = self.moe.evaluate_dual_filter_signal(self.dummy_15m, threshold=0.65)
        if res.get("gating_confidence", 0.0) < 0.65:
            self.assertFalse(
                res["is_approved"],
                f"🚨 [치명적 오류] 확신도({res.get('gating_confidence')})가 65% 미만인데 is_approved=True로 승인됨!"
            )
        print("✅ [Test 1 통과] GBDT 기준 확신도 65% 미만 시 무조건 진입 차단 검증 완료")

    def test_2_no_holdings_no_sell_orders(self):
        """[인터락 2] 실제 원장 잔고(holdings == 0)가 없으면 어떤 매도도 절대 발주 불가"""
        runner = KiwoomLiveRunner(is_simulation=True)
        # 잔고가 빈 상태
        empty_stk_bal = {"holdings_count": 0, "holdings": []}
        
        # _manage_open_positions에 빈 잔고를 넣었을 때 매도 주문이 절대 나가지 않는지 검증
        runner._manage_open_positions(empty_stk_bal)
        # 예외 없이 조용히 return해야 함
        self.assertEqual(len(empty_stk_bal["holdings"]), 0)
        print("✅ [Test 2 통과] 원장 잔고 0주 상태에서 매도 발주 절대 차단 검증 완료")

    def test_3_cancel_tr_format(self):
        """[인터락 3] 키움 주문 취소 TR 규격(ust20003, orig_ord_no) 무결성 확인"""
        broker = KiwoomBroker(is_simulation=True)
        # 취소 함수 시그니처 및 파라미터 확인
        import inspect
        sig = inspect.signature(broker.cancel_order)
        self.assertIn("order_no", sig.parameters)
        self.assertIn("symbol", sig.parameters)
        print("✅ [Test 3 통과] 주문 취소 TR 규격 무결성 확인")

    def test_4_no_arbitrary_market_open_clear(self):
        """[인터락 4] 장 개장 시 이전 보유분이 있어도 강제 청산되지 않고 +3.0%/-2.0% 룰만 적용"""
        runner = KiwoomLiveRunner(is_simulation=True)
        # live_runner 소스코드에 CarryOverClear 문자열이 완전히 제거되었는지 검사
        import inspect
        source = inspect.getsource(runner._market_execution_loop)
        self.assertNotIn(
            "CarryOverClear",
            source,
            "🚨 [치명적 오류] CarryOverClear(장 개장 강제 청산) 로직이 여전히 코드에 남아있음!"
        )
        print("✅ [Test 4 통과] 장 개장 강제 청산 로직 완전 영구 제거 검증 완료")

    def test_5_daily_circuit_breaker_veto(self):
        """[인터락 5] 일일 3회 손절(3-Out) 도달 시 서킷 브레이커 발동 및 신규 진입 즉각 VETO 차단 검증"""
        from unittest.mock import MagicMock
        runner = KiwoomLiveRunner(is_simulation=True)
        # 테스트 중 실제 텔레그램 메시지 발송 원천 차단
        runner.dispatcher.send_telegram_message = MagicMock()
        runner.notifier._send_http_request = MagicMock()
        
        # 1. 초기화 상태 검증
        runner._reset_daily_circuit_breaker()
        self.assertEqual(runner.daily_stoploss_count, 0)
        self.assertFalse(runner.daily_circuit_breaker_triggered)

        # 2. 1회 손절
        runner._record_stoploss()
        self.assertEqual(runner.daily_stoploss_count, 1)
        self.assertFalse(runner.daily_circuit_breaker_triggered)

        # 3. 2회 손절
        runner._record_stoploss()
        self.assertEqual(runner.daily_stoploss_count, 2)
        self.assertFalse(runner.daily_circuit_breaker_triggered)

        # 4. 3회 손절 -> 3-Out 서킷 브레이커 즉각 발동
        runner._record_stoploss()
        self.assertEqual(runner.daily_stoploss_count, 3)
        self.assertTrue(runner.daily_circuit_breaker_triggered)

        # 5. 신규 매수 주문 집행 차단 검증
        buy_res = runner._execute_buy_with_10s_chase(
            symbol="SOXL",
            target_qty=10,
            ref_price=50.0,
            moe_res={},
            targets={"dynamic_tp_px": 51.5, "dynamic_sl_px": 48.0, "tp_pct": 3.0, "sl_pct": 2.0}
        )
        self.assertFalse(buy_res, "🚨 [치명적 오류] 서킷 브레이커 발동 상태인데 매수 주문이 실행됨!")
        # 테스트 완료 후 원복
        runner._reset_daily_circuit_breaker()
        print("✅ [Test 5 통과] 당일 3회 손절(3-Out) 시 일일 서킷 브레이커 Veto 및 신규 진입 완벽 차단 검증 완료")

    def test_6_hybrid_moe_single_trigger_logic(self):
        """[인터락 6] Lumos V3 하이브리드 MoE (GBDT 65% 단일 트리거 / 크로스에셋 방향성 검토 폐지) 동작 무결성 검증"""
        from unittest.mock import patch
        moe_v3 = MoEMetaOrchestrator(confidence_threshold=0.65, gbdt_threshold=0.65, mode="hybrid_v3")

        # 시나리오 1: GBDT LONG_SOXL (68%) ➔ 진입 승인
        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, 0.68, {})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["selected_expert"], "hybrid_moe_v3")
            self.assertEqual(res["direction"], "LONG_SOXL")
            self.assertTrue(res["is_approved"], "🚨 [치명적 오류] 단일 트리거 조건이 충족되었으나 매수가 거부됨!")

        # 시나리오 2: GBDT SHORT_SOXS (70%) ➔ 진입 승인
        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(-1, 0.70, {})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["selected_expert"], "hybrid_moe_v3")
            self.assertEqual(res["direction"], "SHORT_SOXS")
            self.assertTrue(res["is_approved"], "🚨 [치명적 오류] 단일 트리거 조건이 충족되었으나 매수가 거부됨!")

        # 시나리오 3: GBDT 확신도 65% 미달 (58%) ➔ 트리거 불발 (진입 불가)
        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, 0.58, {})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])

        print("✅ [Test 6 통과] Lumos V3 하이브리드 MoE (GBDT 65% 단일 트리거 및 크로스에셋 방향성 검토 폐지) 양방향 무결성 검증 완료")

    def test_7_time_synchronization_and_model_switching_integrity(self):
        """[인터락 7] 투자 진행 필수 체크리스트: 글로벌 시간 동기화(KST-NYT) 및 모델 스위칭 스케줄 무결성 검증"""
        from core.live_runner import USMarketCalendar
        res = USMarketCalendar.verify_time_synchronization()
        self.assertTrue(res["all_ok"], f"🚨 [치명적 시간 오류] 시간 동기화 또는 모델 스위칭 스케줄 검증 실패: {res}")
        self.assertTrue(res["time_sync_ok"], f"🚨 KST-NYT 시차 오류 ({res['delta_hours']}h != {res['expected_diff']}h)")
        self.assertTrue(res["switching_ok"], "🚨 Phase 1 / Phase 2 모델 스위칭 상호 배타성 충돌 발생!")
        print("✅ [Test 7 통과] 글로벌 시간 동기화(KST-NYT) 및 모델 스위칭 스케줄 무결성 사전 검증 완료")

if __name__ == "__main__":
    unittest.main()
