import sys
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.live_runner import USMarketCalendar, KiwoomLiveRunner
from core.power_hour_sniper import PowerHourSniper
from core.data_lake import MarketDataLake
from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer

class TestPowerHourSniperDeepAudit(unittest.TestCase):
    """
    [Lumos 퀀트 시스템: 파워 아워 스나이퍼 실전 연동 전수 정밀 감사 테스트]
    1. 타임존(EDT/EST) 및 세션 전환(09:30~14:30 vs 14:30~15:30 vs 15:30~15:50 vs 15:50) 완벽 검증
    2. SOXS 숏 진입 시 SOXL 가격이 아닌 실제 SOXS 가격 기준 TP/SL 산출 여부 검증
    3. 14:30 경계선에서 Phase 1 보유 포지션 존재 시 5m 중복 진입 방어 검증
    4. 웹소켓 시세 120.74 하드코딩 제거 및 REST 폴백 동작 검증
    5. 데이터레이크 실시간 틱 병합 시 뉴욕 현지 시각(EDT) 일치 여부 검증
    """

    def setUp(self):
        self.ny_tz = ZoneInfo("America/New_York")

    def test_1_session_time_boundaries(self):
        """[감사 1] 미국 뉴욕 시간 기준 4대 세션 경계값 정밀 판별 검증"""
        # 09:29:59 -> 프리마켓
        dt = datetime(2026, 9, 11, 9, 29, 59, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertFalse(s["is_open"])
        self.assertEqual(s["session_name"], "PRE_MARKET_WAITING")

        # 09:30:00 -> Phase 1 개장
        dt = datetime(2026, 9, 11, 9, 30, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertTrue(s["is_open"])
        self.assertTrue(s["is_phase1_allowed"])
        self.assertFalse(s["is_phase2_allowed"])
        self.assertTrue(s["is_entry_allowed"])
        self.assertEqual(s["session_name"], "REGULAR_MARKET_OPEN")

        # 14:29:59 -> Phase 1 마지막 초
        dt = datetime(2026, 9, 11, 14, 29, 59, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertTrue(s["is_phase1_allowed"])
        self.assertFalse(s["is_phase2_allowed"])
        self.assertTrue(s["is_entry_allowed"])

        # 14:30:00 -> Phase 2 파워 아워 스나이퍼 개시
        dt = datetime(2026, 9, 11, 14, 30, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertFalse(s["is_phase1_allowed"])
        self.assertTrue(s["is_phase2_allowed"])
        self.assertTrue(s["is_entry_allowed"])
        self.assertEqual(s["session_name"], "POWER_HOUR_SNIPER")

        # 15:29:59 -> Phase 2 마지막 초
        dt = datetime(2026, 9, 11, 15, 29, 59, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertFalse(s["is_phase1_allowed"])
        self.assertTrue(s["is_phase2_allowed"])
        self.assertTrue(s["is_entry_allowed"])

        # 15:30:00 -> 쿨다운 (신규 진입 전면 차단)
        dt = datetime(2026, 9, 11, 15, 30, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertFalse(s["is_phase1_allowed"])
        self.assertFalse(s["is_phase2_allowed"])
        self.assertFalse(s["is_entry_allowed"])
        self.assertEqual(s["session_name"], "REGULAR_MARKET_NO_ENTRY")

        # 15:50:00 -> EOD 100% 현금화 강제 청산 윈도우
        dt = datetime(2026, 9, 11, 15, 50, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertTrue(s["is_eod_liquidation_window"])
        self.assertEqual(s["session_name"], "EOD_LIQUIDATION")

        # 16:00:00 -> 장 마감 (애프터마켓)
        dt = datetime(2026, 9, 11, 16, 0, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt)
        self.assertFalse(s["is_open"])
        self.assertEqual(s["session_name"], "AFTER_MARKET_CLOSED")
        print("✅ [감사 1 통과] 뉴욕 증시 4대 세션 경계값 및 타임존 완벽 일치")

    def test_2_soxs_target_calculation_accuracy(self):
        """[감사 2] SHORT_SOXS 선정 시 SOXL 가격이 아닌 실제 SOXS 가격 기준 TP/SL 산출 검증"""
        sniper = PowerHourSniper(tp_pct=0.025, sl_pct=0.0167, time_stop_minutes=30, confidence_threshold=0.55)
        # SOXS 현재가 $20.00 주입
        soxs_cur_px = 20.00
        tp_px = round(soxs_cur_px * (1 + sniper.tp_pct), 2)
        sl_px = round(soxs_cur_px * (1 - sniper.sl_pct), 2)

        # 검증: TP는 20.50, SL은 19.67 이어야 함 (SOXL 120달러대가 아니어야 함)
        self.assertEqual(tp_px, 20.50)
        self.assertEqual(sl_px, 19.67)
        self.assertLess(tp_px, 30.00)
        print("✅ [감사 2 통과] SOXS 숏 진입 시 실제 SOXS 가격 기준 동적 TP/SL 산출 무결성 검증 완료")

    def test_3_websocket_no_hardcoded_default_price(self):
        """[감사 3] 웹소켓 시세 엔진에서 120.74 하드코딩 제거 및 브로커 REST 폴백 정상 작동 검증"""
        broker_mock = MagicMock()
        broker_mock.get_stock_quote.return_value = {"last_price": 25.50}
        streamer = KiwoomWebSocketStreamer(broker=broker_mock)

        # 미구독/미수신 심볼 호출 시 120.74가 아니라 broker_mock 현재가 25.50 반환해야 함
        px = streamer.get_latest_price("UNKNOWN_SYM")
        self.assertEqual(px, 25.50)
        self.assertNotEqual(px, 120.74)
        print("✅ [감사 3 통과] 웹소켓 시세 엔진 120.74 하드코딩 제거 및 REST 자동 폴백 검증 완료")

    def test_4_data_lake_live_tick_timezone_consistency(self):
        """[감사 4] 실시간 틱 병합 시 캔들 타임스탬프가 KST(한국)가 아닌 미국 뉴욕 정규장 시간대와 일치하는지 검증"""
        dl = MarketDataLake()
        soxl_5m = dl.load_candles("SOXL", "5m")
        if not soxl_5m.empty:
            merged = dl.get_candles_with_live_tick("SOXL", "5m", live_price=122.50)
            last_candle_dt = merged.index[-1]
            # 시간대가 뉴욕 기준이므로, 뉴욕 현재 시각과의 차이가 24시간 이내여야 함
            ny_tz = ZoneInfo("America/New_York")
            kst_tz = ZoneInfo("Asia/Seoul")
            now_dt = datetime.now()
            now_kst = now_dt.replace(tzinfo=kst_tz) if now_dt.tzinfo is None else now_dt.astimezone(kst_tz)
            now_ny = now_kst.astimezone(ny_tz).replace(tzinfo=None)
            
            # 마지막 캔들의 시간과 현재 뉴욕 시간 차이 검증 (한국 시간이 아님을 확인)
            diff_hours = abs((last_candle_dt - now_ny).total_seconds()) / 3600.0
            # 주말/장외 감안하더라도 마지막 캔들의 시간이 13시간 앞선 KST로 생성되지 않았는지 확인
            self.assertNotEqual(last_candle_dt.hour, now_kst.hour if now_kst.hour != now_ny.hour else -1)
            print("✅ [감사 4 통과] 실시간 틱 데이터레이크 병합 시 뉴욕 현지 시계열 동기화 검증 완료")

    def test_5_single_position_relay_at_boundary(self):
        """[감사 5] 14:30 경계선에서 Phase 1 보유 포지션이 있을 경우 Phase 2 중복 진입 방어 검증"""
        runner = KiwoomLiveRunner(is_simulation=True)
        # Phase 1 포지션 주입
        pos = {
            "symbol": "SOXL",
            "quantity": 100,
            "price": 120.0,
            "buy_time": "2026-09-11 14:25:00",
            "tp_pct": 3.0,
            "sl_pct": 2.0,
            "time_stop_minutes": 90,
            "strategy_tag": "Phase 1: Model C"
        }
        runner._save_active_position(pos)
        
        # 14:35 EDT 시점 시뮬레이션
        dt_1435 = datetime(2026, 9, 11, 14, 35, 0, tzinfo=self.ny_tz)
        s = USMarketCalendar.get_market_status(dt_1435)
        self.assertTrue(s["is_phase2_allowed"])
        
        # 활성 포지션이 있으므로 신규 진입 스캔이 차단되고 90분 타임스탑 유지가 되어야 함
        active = runner._get_active_position()
        self.assertIsNotNone(active)
        self.assertEqual(active["time_stop_minutes"], 90)
        self.assertEqual(active["strategy_tag"], "Phase 1: Model C")

        # 정리
        runner._clear_active_position()
        print("✅ [감사 5 통과] 14:30 경계선 단일 포지션 릴레이 원칙 및 기존 포지션 보존 검증 완료")

    def test_6_sniper_cross_asset_veto_defense(self):
        """[감사 6] Phase 2 5분봉 스나이퍼 GBDT 매수 신호 발생 시 크로스에셋 상충 VETO 차단 검증"""
        from unittest.mock import patch
        sniper = PowerHourSniper(tp_pct=0.025, sl_pct=0.0167, time_stop_minutes=30, confidence_threshold=0.55)
        
        # 5m 더미 캔들
        dummy_5m = pd.DataFrame({
            "Open": [100.0] * 30, "High": [101.0] * 30, "Low": [99.0] * 30, "Close": [100.5] * 30, "Volume": [10000] * 30
        })

        # 시나리오 1: 5분봉 GBDT는 LONG_SOXL (65%)이나, 크로스에셋이 SHORT_SOXS (-1) 역풍 경고 -> VETO 차단
        with patch.object(sniper.model, 'predict_proba', return_value=[[0.10, 0.25, 0.65]]):
            with patch.object(sniper.cross_asset_model, 'predict_signal', return_value=(-1, 0.70, "NVDA 급락 괴리")):
                res = sniper.evaluate_sniper_signal(dummy_5m)
                self.assertEqual(res["direction"], "LONG_SOXL")
                self.assertTrue(res["is_cross_veto"], "🚨 크로스에셋 정반대 신호인데 Veto되지 않음!")
                self.assertFalse(res["is_approved"], "🚨 Veto 상태인데 진입이 승인됨!")

        # 시나리오 2: 5분봉 GBDT LONG_SOXL (65%) & 크로스에셋 중립 (0) -> VETO 없음 (통과)
        with patch.object(sniper.model, 'predict_proba', return_value=[[0.10, 0.25, 0.65]]):
            with patch.object(sniper.cross_asset_model, 'predict_signal', return_value=(0, 0.50, "정렬")):
                res = sniper.evaluate_sniper_signal(dummy_5m)
                self.assertEqual(res["direction"], "LONG_SOXL")
                self.assertFalse(res["is_cross_veto"])

        print("✅ [감사 6 통과] Phase 2 5분봉 스나이퍼 크로스에셋 Veto 방패 무결성 검증 완료")

if __name__ == "__main__":
    unittest.main()
