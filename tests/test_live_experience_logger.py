import os
import sys
import gc
import time
import shutil
import unittest
import tempfile
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.live_experience_logger import LiveExperienceLogger, CSV_COLUMNS

class TestLiveExperienceLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir_path = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir_path / "test_exp.db"
        self.csv_path = self.temp_dir_path / "test_trades.csv"
        # Reset singleton for isolated test instance
        LiveExperienceLogger._instance = None
        self.logger = LiveExperienceLogger(db_path=self.db_path, csv_path=self.csv_path)

    def tearDown(self):
        LiveExperienceLogger._instance = None
        gc.collect()
        time.sleep(0.1)
        shutil.rmtree(self.temp_dir_path, ignore_errors=True)

    def test_record_trade_dual_storage(self):
        """[검증 1] 거래 기록 시 CSV와 SQLite DB에 완벽히 동시 적재되는지 검증"""
        sample_trade = {
            "mode": "VIRTUAL",
            "symbol": "SOXL",
            "entry_time": "2026-09-14 09:45:00",
            "exit_time": "2026-09-14 10:15:00",
            "intended_entry_price": 120.00,
            "actual_entry_price": 120.03,
            "intended_exit_price": 123.60,
            "actual_exit_price": 123.63,
            "quantity": 100,
            "pnl_pct": 3.00,
            "pnl_usd": 360.00,
            "pnl_krw": 486000,
            "exit_reason": "TAKE_PROFIT_3.0%",
            "mfe_pct": 3.25,
            "mae_pct": -0.45,
            "gbdt_confidence": 0.6540,
            "cross_dir": "LONG",
            "features": {
                "VWAP_Diff": 0.012,
                "RSI_14": 58.4,
                "ATR_14": 2.15,
                "Volume_Z": 1.84
            }
        }

        trade_id = self.logger.record_trade(sample_trade)
        self.assertTrue(trade_id.startswith("LIVE_"))

        # 1. CSV 파일 검증
        self.assertTrue(self.csv_path.exists())
        df = pd.read_csv(self.csv_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["symbol"], "SOXL")
        self.assertEqual(df.iloc[0]["quantity"], 100)
        self.assertAlmostEqual(df.iloc[0]["entry_slippage_usd"], 0.03, places=2)
        self.assertAlmostEqual(df.iloc[0]["mfe_pct"], 3.25, places=2)
        self.assertAlmostEqual(df.iloc[0]["mae_pct"], -0.45, places=2)
        self.assertIn("VWAP_Diff", df.iloc[0]["features_json"])

        # 2. SQLite DB 검증
        conn = self.logger._get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM live_trades WHERE trade_id = ?", (trade_id,))
            row = c.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["symbol"], "SOXL")
            self.assertEqual(row["exit_reason"], "TAKE_PROFIT_3.0%")
            self.assertAlmostEqual(row["gbdt_confidence"], 0.6540, places=4)
        finally:
            conn.close()

    def test_record_order_event(self):
        """[검증 2] 3초 타임아웃 및 주문 체이싱 이벤트가 SQLite에 올바르게 기록되는지 검증"""
        self.logger.record_order_event(
            symbol="SOXL",
            action="BUY",
            attempt=1,
            order_no="12345",
            price=120.03,
            quantity=100,
            status="UNFILLED_TIMEOUT",
            note="3초 타임아웃 발생으로 주문 취소 후 재발주"
        )

        conn = self.logger._get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM live_order_events WHERE order_no = '12345'")
            row = c.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "UNFILLED_TIMEOUT")
            self.assertEqual(row["attempt"], 1)
        finally:
            conn.close()

    def test_record_error_event(self):
        """[검증 3] 증권사 API 에러 전문이 SQLite에 정확히 보존되는지 검증"""
        self.logger.record_error_event(
            source="KiwoomBroker",
            error_type="HTTP_ERROR",
            error_code="429",
            message="요청 한도 초과",
            payload={"cano": "61112456", "api_id": "ust20000"}
        )

        conn = self.logger._get_connection()
        try:
            c = conn.cursor()
            c.execute("SELECT * FROM live_system_errors WHERE error_code = '429'")
            row = c.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["error_type"], "HTTP_ERROR")
            self.assertIn("ust20000", row["payload_json"])
        finally:
            conn.close()

    def test_summary_stats(self):
        """[검증 4] 누적 승률 및 슬리피지 통계 산출 정확성 검증"""
        self.logger.record_trade({
            "symbol": "SOXL",
            "quantity": 10,
            "actual_entry_price": 100.0,
            "intended_entry_price": 99.98,
            "actual_exit_price": 103.0,
            "pnl_pct": 3.0,
            "pnl_usd": 30.0,
            "pnl_krw": 40500,
            "hold_minutes": 25.0
        })
        self.logger.record_trade({
            "symbol": "SOXS",
            "quantity": 20,
            "actual_entry_price": 50.0,
            "intended_entry_price": 50.02,
            "actual_exit_price": 49.0,
            "pnl_pct": -2.0,
            "pnl_usd": -20.0,
            "pnl_krw": -27000,
            "hold_minutes": 15.0
        })

        stats = self.logger.get_summary_stats()
        self.assertEqual(stats["total_trades"], 2)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["win_rate_pct"], 50.0)
        self.assertEqual(stats["total_pnl_usd"], 10.0)
        self.assertEqual(stats["total_pnl_krw"], 13500)
        self.assertEqual(stats["avg_hold_minutes"], 20.0)

if __name__ == "__main__":
    unittest.main()
