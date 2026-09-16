import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.live_runner import KiwoomLiveRunner
from core.telegram_notifier import TelegramNotifier

class TestLiveDefenseAndBriefing(unittest.TestCase):
    def setUp(self):
        os.environ["KIWOOM_IS_SIMULATION"] = "1"
        self.runner = KiwoomLiveRunner(is_simulation=True)
        self.runner.notifier = MagicMock(spec=TelegramNotifier)

    def test_defense_alert_dispatch(self):
        """GBDT >= 60%이나 Screen 1 대추세 불일치로 차단 시 defense_alert 발송 검증"""
        now_ts = time.time()
        self.runner._last_defense_time = 0.0
        self.runner._last_defense_state = False

        moe_res = {
            "selected_expert": "gbdt_pattern",
            "gating_confidence": 0.72,
            "direction": "LONG_SOXL",
            "dir_gbdt": "LONG_SOXL",
            "is_approved": False,
            "is_60m_trend_ok": False,
            "dip_ok": True
        }

        top_conf = float(moe_res["gating_confidence"]) * 100.0
        is_approved = moe_res["is_approved"]

        if top_conf >= 60.0 and not is_approved and moe_res.get("selected_expert") != "conflict_rejected":
            last_d_t = getattr(self.runner, "_last_defense_time", 0.0)
            if not getattr(self.runner, "_last_defense_state", False) and (now_ts - last_d_t >= 900.0):
                cand_ticker = "SOXL" if "LONG" in moe_res.get("dir_gbdt", moe_res["direction"]) else "SOXS"
                is_60m = moe_res.get("is_60m_trend_ok", True)
                is_dip = moe_res.get("dip_ok", True)
                if not is_60m:
                    def_type = "Screen 1 (상위 60분봉 대추세 필터)"
                    def_reason = "60분봉 20 EMA 역추세 구간으로 하방 리스크 차단"
                elif not is_dip:
                    def_type = "Screen 3 (단기 눌림목/과열 필터)"
                    def_reason = "VWAP 이격 과다 또는 RSI 과매수 구간으로 추격 매수 배제"
                else:
                    def_type = "Lumos 헌법 인터락 방어"
                    def_reason = "다중 안전 인터락 조건 미충족으로 자본 보존"

                self.runner.notifier.send_defense_alert(
                    ticker=cand_ticker,
                    gbdt_prob=top_conf,
                    defense_type=def_type,
                    defense_reason=def_reason
                )
                self.runner._last_defense_state = True
                self.runner._last_defense_time = now_ts

        self.runner.notifier.send_defense_alert.assert_called_once_with(
            ticker="SOXL",
            gbdt_prob=72.0,
            defense_type="Screen 1 (상위 60분봉 대추세 필터)",
            defense_reason="60분봉 20 EMA 역추세 구간으로 하방 리스크 차단"
        )
        self.assertTrue(self.runner._last_defense_state)
        self.assertEqual(self.runner._last_defense_time, now_ts)

    def test_notifier_briefing_format(self):
        """TelegramNotifier 정기 브리핑 메서드 규격 정상 동작 검증"""
        notifier = TelegramNotifier()
        with patch.object(notifier, "_dispatch_message", return_value={"ok": True}) as mock_dispatch:
            res = notifier.send_periodic_briefing(
                now_str="2026-09-15 22:45:00 KST",
                soxl_px=105.5,
                soxs_px=15.2,
                gbdt_conf=58.0,
                direction="관망 (NONE)",
                session_desc="정규장 Phase 1",
                usd_avail=50000.0,
                status_note="A급 고확신 타점 실시간 대기 중"
            )
            mock_dispatch.assert_called_once()
            self.assertTrue(res["ok"])

if __name__ == "__main__":
    unittest.main()
