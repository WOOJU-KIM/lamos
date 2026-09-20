import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from core.system_logger import system_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

class StateTracker:
    def __init__(self, dispatcher=None):
        self._active_position = None
        self._daily_cb_file = DATA_DIR / "daily_cb_state.json"
        self.daily_stoploss_count = 0
        self.daily_circuit_breaker_triggered = False
        self.dispatcher = dispatcher
        self._load_daily_cb_state()

    def _save_active_position(self, pos_data: Dict[str, Any]):
        """???? ? ???? ???? ? ???"""
        self._active_position = pos_data
        try:
            pos_file = DATA_DIR / "active_position.json"
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump(pos_data, f, ensure_ascii=False, indent=2)
            system_logger.info(f"? [???? ????] {pos_data.get('symbol')} {pos_data.get('quantity')}?@ ${pos_data.get('price')} (?: {pos_data.get('buy_time')})")
        except Exception as e:
            system_logger.warn(f"???? ????: {e}")

    def _get_active_position(self) -> Optional[Dict[str, Any]]:
        if self._active_position:
            return self._active_position
        pos_file = DATA_DIR / "active_position.json"
        if pos_file.exists():
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    self._active_position = json.load(f)
                    return self._active_position
            except Exception:
                pass
        return None

    def _clear_active_position(self):
        self._active_position = None
        try:
            pos_file = DATA_DIR / "active_position.json"
            if pos_file.exists():
                pos_file.unlink()
            system_logger.info("? [???? ?? 100% ? ??? ?")
        except Exception as e:
            system_logger.warn(f"???? ?? ?: {e}")

    def _get_current_ny_date(self) -> str:
        ny_tz = ZoneInfo("America/New_York")
        return datetime.now().astimezone().astimezone(ny_tz).strftime("%Y-%m-%d")

    def _load_daily_cb_state(self):
        today_ny = self._get_current_ny_date()
        if self._daily_cb_file.exists():
            try:
                with open(self._daily_cb_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == today_ny:
                    self.daily_stoploss_count = int(data.get("stoploss_count", 0))
                    self.daily_circuit_breaker_triggered = bool(data.get("circuit_breaker_triggered", False))
                    system_logger.info(f"??[? ? ? ] ?: {today_ny} | ? ?: {self.daily_stoploss_count}/3??|  ?: {self.daily_circuit_breaker_triggered}")
                    return
            except Exception as e:
                system_logger.info(f"? ? ?  ?: {e}")
        self.daily_stoploss_count = 0
        self.daily_circuit_breaker_triggered = False

    def _save_daily_cb_state(self):
        today_ny = self._get_current_ny_date()
        data = {
            "date": today_ny,
            "stoploss_count": self.daily_stoploss_count,
            "circuit_breaker_triggered": self.daily_circuit_breaker_triggered,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(self._daily_cb_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            system_logger.warn(f"? ? ? ????: {e}")

    def _reset_daily_circuit_breaker(self):
        self.daily_stoploss_count = 0
        self.daily_circuit_breaker_triggered = False
        self._save_daily_cb_state()
        system_logger.info("? [? ? ? ?? ???(09:30 NYT): daily_stoploss_count = 0, ?   ?")

    def _record_stoploss(self):
        """
#         - '-2.0% ?? ?  ??daily_stoploss_count 1 ? (?/?????)
        """
        self.daily_stoploss_count += 1
        system_logger.warn(f"? [? ?????] ? ?: {self.daily_stoploss_count}/3??")
        system_logger.log("RISK", "CircuitBreaker", f"? ? -2.0% ?? ({self.daily_stoploss_count}/3??")

        if self.daily_stoploss_count >= 3:
            self.daily_circuit_breaker_triggered = True
            system_logger.error(f"? [? ? ? ] ? ? 3???(3-Out) ??? ?  ? ")
            system_logger.log("RISK", "CircuitBreaker", "? ? ? ? : ? ????(3-Out Veto)")

            msg = f"""🚨 **일일 3-Out 서킷 브레이커 발동**\n- 금일 손절 횟수: {self.daily_stoploss_count}/3\n- 조치: 신규 매수 전면 차단 (VETO)\n- 현금: 100% 보존"""

            msg = "Circuit Breaker Triggered"
            try:
                self.dispatcher.send_telegram_message(msg)
            except Exception as te:
                system_logger.warn(f"? ? ?  ?: {te}")

        self._save_daily_cb_state()
