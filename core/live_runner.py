import config
import os
# import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows  utf-8 ?
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    INITIAL_CAPITAL_KRW,
    DATA_DIR
)
from core.kiwoom_broker import KiwoomBroker
from agents.dispatcher_agent import DispatcherAgent
from core.telegram_controller import TelegramController
from core.state_hub import StateHub
from core.data_lake import MarketDataLake
from core.shadow_sandbox import ShadowSandboxEngine
from core.circuit_breaker import CircuitBreakerEngine
from core.system_logger import system_logger
from core.moe_orchestrator import MoEMetaOrchestrator
from core.live_experience_logger import LiveExperienceLogger

logger = logging.getLogger("LiveRunner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][LiveRunner] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class KRMarketCalendar:
    """
    국내장(KOSPI/KOSDAQ) 영업시간 캘린더
    - 정규장 오픈: 09:30 KST (모의투자 등을 고려하여 09:00 대신 09:30으로 늦춰서 안전하게 매매하는 경우도 있지만, KOSPI 기준 정규는 09:00)
    - 우리는 09:00 ~ 15:30으로 설정.
    """

    @staticmethod
    def get_market_status(now_dt: Optional[datetime]=None) -> Dict[str, Any]:
        if now_dt is None:
            now_dt = datetime.now()
        kst_tz = ZoneInfo('Asia/Seoul')
        
        if now_dt.tzinfo is None:
            now_kst = now_dt.replace(tzinfo=kst_tz)
        else:
            now_kst = now_dt.astimezone(kst_tz)
            
        weekday_kst = now_kst.weekday()
        kst_time = now_kst.time()
        is_weekend = weekday_kst >= 5
        
        market_open_time = dtime(9, 0)
        trading_cutoff_time = dtime(15, 15)
        eod_liquidation_time = dtime(15, 15)
        market_close_time = dtime(15, 30)
        
        is_regular_hours = not is_weekend and market_open_time <= kst_time < market_close_time
        is_phase2_allowed = False
        is_trading_allowed = not is_weekend and market_open_time <= kst_time < trading_cutoff_time
        is_entry_allowed = is_trading_allowed
        is_eod_liquidation_window = not is_weekend and eod_liquidation_time <= kst_time < market_close_time
        
        if is_regular_hours:
            next_open_kst = None
            time_until_open_str = '   '
        else:
            target_kst_date = now_kst.date()
            if is_weekend:
                days_ahead = 7 - weekday_kst
                target_kst_date += timedelta(days=days_ahead)
            elif kst_time >= market_close_time:
                if weekday_kst == 4:
                    target_kst_date += timedelta(days=3)
                else:
                    target_kst_date += timedelta(days=1)
            next_open_kst = datetime.combine(target_kst_date, market_open_time, tzinfo=kst_tz)
            delta = next_open_kst - now_kst
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            mins, secs = divmod(remainder, 60)
            time_until_open_str = f'{hours}hours {mins}mins'
            
        if is_weekend:
            day_name_kr = '주말'
            session_name = 'WEEKEND_CLOSED'
            status_desc = f'휴장일 ({day_name_kr})'
        elif not is_regular_hours:
            if kst_time < market_open_time:
                session_name = 'PRE_MARKET_WAITING'
                status_desc = '개장 대기'
            else:
                session_name = 'AFTER_MARKET_CLOSED'
                status_desc = '장 마감'
        elif is_eod_liquidation_window:
            session_name = 'EOD_LIQUIDATION'
            status_desc = '마감전 100% 현금화'
        elif is_trading_allowed:
            session_name = 'REGULAR_MARKET_OPEN'
            status_desc = '정규장 (15m Model C)'
        else:
            session_name = 'REGULAR_MARKET_NO_ENTRY'
            status_desc = '신규 진입 금지'
            
        return {
            'is_open': is_regular_hours, 
            'is_entry_allowed': is_entry_allowed, 
            'is_trading_allowed': is_trading_allowed, 
            'is_phase2_allowed': is_phase2_allowed, 
            'is_weekend': is_weekend, 
            'is_eod_liquidation_window': is_eod_liquidation_window, 
            'session_name': session_name, 
            'status_desc': status_desc, 
            'now_kst_str': now_kst.strftime('%Y-%m-%d %H:%M:%S KST'), 
            'next_open_kst_str': next_open_kst.strftime('%Y-%m-%d (%a) %H:%M KST') if next_open_kst else 'Open', 
            'time_until_open_str': time_until_open_str
        }

    @staticmethod
    def verify_time_synchronization() -> Dict[str, Any]:
        kst_tz = ZoneInfo('Asia/Seoul')
        now_local = datetime.now().astimezone()
        now_kst = now_local.astimezone(kst_tz)
        
        # OS 시간대가 Asia/Seoul과 일치하는지 확인
        delta_hours = round((now_local.utcoffset().total_seconds() - now_kst.utcoffset().total_seconds()) / 3600.0, 1)
        time_sync_ok = delta_hours == 0.0
        
        test_weekday = now_kst.date() - timedelta(days=now_kst.weekday())
        test_p1 = datetime.combine(test_weekday, dtime(11, 0), tzinfo=kst_tz)
        test_eod = datetime.combine(test_weekday, dtime(15, 20), tzinfo=kst_tz)
        
        s_p1 = KRMarketCalendar.get_market_status(test_p1)
        s_eod = KRMarketCalendar.get_market_status(test_eod)
        
        switching_ok = s_p1['is_entry_allowed'] and (not s_p1['is_eod_liquidation_window']) and s_eod['is_eod_liquidation_window'] and (not s_eod['is_entry_allowed'])
        all_ok = bool(time_sync_ok and switching_ok)
        
        return {
            'all_ok': all_ok, 
            'time_sync_ok': time_sync_ok, 
            'switching_ok': switching_ok, 
            'now_kst_str': now_kst.strftime('%Y-%m-%d %H:%M:%S KST')
        }

class KiwoomLiveRunner:
    """
    [??(Kiwoom) ????????  ??? ???]
    1.  ?? ? 100% ?: ? //?/ ?????? ???
    2.   ?AI ???? 100% ? (  ? )
    3. 3????????&   (: 1????/ : 100% ? ? ?)
    4. ??? ???  0.5?/ ? 0.3??  ?HTTP 429 ? ???
    """
    # ??[?????????? (Hard Rule #4: 3.0?????]
    ORDER_TIMEOUT_BUY_SEC: float = 3.0   #  ?????(3?
    ORDER_TIMEOUT_SELL_SEC: float = 3.0  #  ?????(3?

    def __init__(self, is_simulation: Optional[bool] = None):
        from core.state_tracker import StateTracker
        from core.order_execution_engine import OrderExecutionEngine
        from core.briefing_manager import BriefingManager
        self.state_tracker = StateTracker(dispatcher=self.dispatcher if hasattr(self, 'dispatcher') else None)
        

        env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
        sim_flag = (env_sim == "1" or env_sim.lower() == "true") if is_simulation is None else bool(is_simulation)
        self.broker = KiwoomBroker(is_simulation=sim_flag)
        self.dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.controller = TelegramController(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        from core.telegram_notifier import TelegramNotifier
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.state_hub = StateHub()
        self.data_lake = MarketDataLake()
        self.shadow_sandbox = ShadowSandboxEngine()
        self.circuit_breaker = CircuitBreakerEngine()
        self.moe_orchestrator = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD)
        self.enable_5m_sniper = False  # ? 15 ? ?? (? 77.55%, PF 3.98  / 5 ?? ? )
#         #self.sniper_engine = PowerHourSniper(
#             tp_pct=0.025,
#             sl_pct=0.0167,
#             time_stop_minutes=30,
#             confidence_threshold=0.55
#         )
        self.experience_logger = LiveExperienceLogger()

        # ??[?????????? (Hard Rule #4: 3.0????]
        self.order_timeout_buy_sec: float = float(os.getenv("LUMOS_ORDER_TIMEOUT_BUY_SEC", str(self.ORDER_TIMEOUT_BUY_SEC)))
        self.order_timeout_sell_sec: float = float(os.getenv("LUMOS_ORDER_TIMEOUT_SELL_SEC", str(self.ORDER_TIMEOUT_SELL_SEC)))
        
        # ??????WebSocket) ???? ? ?
        from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
        self.ws_streamer = KiwoomWebSocketStreamer(broker=self.broker)
        self.ws_streamer.register_callback(self._on_websocket_tick)
        self.ws_streamer.register_execution_callback(self._on_websocket_execution)
        self.briefing_manager = BriefingManager(dispatcher=self.dispatcher if hasattr(self, "dispatcher") else None)
        self.order_engine = OrderExecutionEngine(broker=self.broker if hasattr(self, 'broker') else None, dispatcher=self.dispatcher if hasattr(self, 'dispatcher') else None, state_tracker=self.state_tracker, ws_streamer=self.ws_streamer)

        self.is_running = False
        self._last_market_session = None
        self.state_tracker._active_position: Optional[Dict[str, Any]] = None  # ? ??????
        self._is_order_in_progress = False                      #   ???(AI ? 100% ?)
        self._order_lock = threading.Lock()
        self._order_fills: Dict[str, int] = {}                  # ????  ? {order_no: filled_qty}
        self._eod_liquidation_done = False

        # ??[? ? ?: Daily Circuit Breaker 3-Out Veto]
        self.state_tracker.daily_stoploss_count: int = 0
        self.state_tracker.daily_circuit_breaker_triggered: bool = False
        self.state_tracker._daily_cb_file = DATA_DIR / "daily_circuit_breaker_state.json"
        self.state_tracker._load_daily_cb_state()

        # ??[ ? ?15?? ???  ??
        self._last_veto_time: float = 0.0
        self._last_veto_state: bool = False
        self._last_defense_time: float = 0.0
        self._last_defense_state: bool = False
        self._last_briefing_time: float = 0.0
        self._last_moe_res: Optional[Dict[str, Any]] = None









    def _on_websocket_execution(self, exec_data: Dict[str, Any]):
        """
        """
        system_logger.info(f"? [WebSocket ?? ? ?]: {exec_data}")
        if not exec_data:
            return

        #  values  ?   ?
        vals = exec_data.get("values", {}) if isinstance(exec_data.get("values"), dict) else {}
        
        def _get_val(*keys):
            for k in keys:
                if k in exec_data and exec_data[k] is not None and str(exec_data[k]).strip() != "":
                    return exec_data[k]
                if k in vals and vals[k] is not None and str(vals[k]).strip() != "":
                    return vals[k]
            return None

        ord_no_raw = _get_val("order_no", "ord_no", "odno", "orgn_odno", "ord_no1", "1")
        ord_no = str(ord_no_raw).strip() if ord_no_raw is not None else ""
        if not ord_no:
            system_logger.info(f" ?  ? ?: {exec_data}")
            return

        def _to_int(val, default=0):
            try:
                if val is None or str(val).strip() == "":
                    return default
                return int(float(str(val).replace(",", "").strip()))
            except Exception:
                return default

        tot_che_qty = _to_int(_get_val(
            "tot_che_qty", "ft_tot_ccld_qty", "acc_filled_qty", "tot_qty", "tot_ccld_qty", "acc_qty", "14"
        ))
        che_qty = _to_int(_get_val(
            "filled_qty", "che_qty", "ft_ccld_qty", "qty", "ccld_qty", "che_qty1", "11"
        ))
        nccs_raw = _get_val("nccs_qty", "ft_rem_qty", "unfilled_qty", "rem_qty", "ord_rem_qty", "13")

        # ?  ? ???
        if tot_che_qty > 0:
            self._order_fills[ord_no] = tot_che_qty
            system_logger.info(f"? [ ? ? ]  #{ord_no} ??? : {tot_che_qty}?")
        elif che_qty > 0:
            self._order_fills[ord_no] = self._order_fills.get(ord_no, 0) + che_qty
            system_logger.info(f"? [ ? ??]  #{ord_no} ???: +{che_qty}?(?{self._order_fills[ord_no]}?")

        # ?0 ? ??100%  ?
        if nccs_raw is not None and _to_int(nccs_raw) == 0:
            current_val = self._order_fills.get(ord_no, che_qty or 1)
            self._order_fills[ord_no] = max(current_val, 1)
            system_logger.info(f"? [? 100%  ?]  #{ord_no} ???? 0?? (?: {self._order_fills[ord_no]}?")


    def _on_websocket_tick(self, symbol: str, price: float, extra: Dict[str, Any]):
        """
        """
        if self._is_order_in_progress:
            return

        try:
            mkt = KRMarketCalendar.get_market_status()
            if not mkt.get("is_open"):
                return

            active_pos = self.state_tracker._get_active_position()
            if not active_pos or active_pos.get("symbol") != symbol:
                return

            qty = int(active_pos.get("quantity", 0))
            buy_px = float(active_pos.get("price", 0.0))
            buy_time_str = active_pos.get("buy_time")

            if qty <= 0 or buy_px <= 0:
                return

            #  ?/? ?? (MFE / MAE ? ???
            cur_hi = active_pos.get("high_price_during_hold", price)
            cur_lo = active_pos.get("low_price_during_hold", price)
            if price > cur_hi:
                active_pos["high_price_during_hold"] = price
            if price < cur_lo:
                active_pos["low_price_during_hold"] = price

            pnl_pct = (price - buy_px) / buy_px

            tp_threshold = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos.get("tp_pct") else config.MAX_TP_PCT
            sl_threshold = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos.get("sl_pct") else config.SL_MIN_PCT
            time_stop_minutes = float(active_pos.get("time_stop_minutes", config.TIME_STOP_MINUTES))

            # 1.  ?
            if pnl_pct >= tp_threshold:
                gain_pct = pnl_pct * 100
                system_logger.info(f"? [??? ????] {symbol} {qty}? ?  (?? +{gain_pct:.2f}%)")
                self.order_engine._execute_sell_with_10s_chase(
                    symbol=symbol,
                    quantity=qty,
                    reason_desc=f"?  ? (+{gain_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=price,
                    is_stoploss=False
                )

            # 2. ??
            elif pnl_pct <= -sl_threshold:
                loss_pct = abs(pnl_pct * 100)
                system_logger.info(f"? [??? ????] {symbol} {qty}? ?  (?? -{loss_pct:.2f}%)")
                self.order_engine._execute_sell_with_10s_chase(
                    symbol=symbol,
                    quantity=qty,
                    reason_desc=f"? ?? (-{loss_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=price,
                    is_stoploss=True
                )

            # 3. ????
            elif buy_time_str:
                try:
                    buy_dt = datetime.strptime(buy_time_str, "%Y-%m-%d %H:%M:%S")
                    elapsed_min = (datetime.now() - buy_dt).total_seconds() / 60.0
                    if elapsed_min >= time_stop_minutes:
                        cur_pnl = pnl_pct * 100
                        system_logger.info(f"??[??{time_stop_minutes:.0f}??????] {symbol} {qty}??  (: {elapsed_min:.0f}?| ?? {cur_pnl:+.2f}%)")
                        self.order_engine._execute_sell_with_10s_chase(
                            symbol=symbol,
                            quantity=qty,
                            reason_desc=f"??{time_stop_minutes:.0f}?????? ({elapsed_min:.0f}?)",
                            buy_px=buy_px,
                            cur_px=price,
                            is_stoploss=False
                        )
                except Exception as te:
                    system_logger.info(f"???? ?: {te}")

        except Exception as e:
            system_logger.info(f"WS ?? ?: {e}")

    def _manage_open_positions(self, stk_bal: Dict[str, Any], realtime_px_override: Optional[Dict[str, float]] = None):
        """
        """
        holdings = stk_bal.get("holdings", [])
        if not holdings:
            return

        active_pos = self.state_tracker._get_active_position()
        tp_max_pct = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos else config.MAX_TP_PCT
        sl_initial_pct = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos else config.SL_MIN_PCT

        for h in holdings:
            sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
            qty = int(float(str(h.get("quantity") or h.get("poss_qty") or h.get("sell_alowq") or 0).replace(",", "")))
            buy_px = float(str(h.get("purchase_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))

            if not sym or qty <= 0 or buy_px <= 0:
                continue

            if realtime_px_override and sym in realtime_px_override:
                cur_px = float(realtime_px_override[sym])
            else:
                quote = self.broker.get_stock_quote(sym)
                cur_px = float(quote.get("last_price", buy_px))

            if cur_px <= 0:
                cur_px = buy_px

            peak_high = buy_px
            if active_pos:
                peak_high = float(active_pos.get("high_price_during_hold", buy_px))
                cur_lo = float(active_pos.get("low_price_during_hold", cur_px))
                if cur_px > peak_high:
                    peak_high = cur_px
                    active_pos["high_price_during_hold"] = peak_high
                if cur_px < cur_lo:
                    active_pos["low_price_during_hold"] = cur_px

            is_trailing_active = False
            current_sl_px = buy_px * (1.0 - sl_initial_pct)
            trailing_trigger_px = buy_px * (1.0 + config.TRAILING_TRIGGER_PCT)
            
            if peak_high >= trailing_trigger_px:
                is_trailing_active = True
                
            if is_trailing_active:
                t = peak_high * (1.0 - config.TRAILING_DROP_PCT)
                if t > current_sl_px:
                    current_sl_px = t
            
            if active_pos:
                active_pos["dynamic_sl_px"] = current_sl_px
                self.state_tracker._save_active_position(active_pos)
                
            tp_max_px = buy_px * (1.0 + tp_max_pct)

            if cur_px >= tp_max_px:
                gain_pct = (cur_px / buy_px - 1.0) * 100.0
                self.order_engine._execute_sell_with_10s_chase(
                    symbol=sym,
                    quantity=qty,
                    reason_desc=f"?  ? ? (+{gain_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=cur_px,
                    is_stoploss=False
                )
            elif cur_px <= current_sl_px:
                loss_pct = (cur_px / buy_px - 1.0) * 100.0
                reason = f"??  ?? ? ({loss_pct:.2f}%)" if is_trailing_active else f"? ATR ? ? ({loss_pct:.2f}%)"
                self.order_engine._execute_sell_with_10s_chase(
                    symbol=sym,
                    quantity=qty,
                    reason_desc=reason,
                    buy_px=buy_px,
                    cur_px=cur_px,
                    is_stoploss=True
                )

    def _sync_real_ledger_entry(self, symbol: str, default_price: float, default_qty: int) -> Tuple[float, int]:
        """
        """
        try:
            stk_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal.get("ok"):
                for h in stk_bal.get("holdings", []):
                    sym_h = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
                    if sym_h == symbol.upper().strip():
                        p_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        b_px = float(str(h.get("purchase_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))
                        if b_px > 0:
                            system_logger.info(f"? [??? ?? ??? ??: ${default_price:.2f} ??? ? ??: ${b_px:.2f} (? ?: {p_qty}?")
                            return round(b_px, 2), (p_qty if p_qty > 0 else default_qty)
        except Exception as e:
            system_logger.warn(f"? ? ?? ???? (?? fallback ??): {e}")
        return default_price, default_qty


    # ??[???????? ]

    def _market_execution_loop(self):
        system_logger.info("?? [?  ?????  ? ? (: 15?...")
        system_logger.log("INFO", "LiveRunner", "???  ??????")
        
        while True:
            try:
                mkt = KRMarketCalendar.get_market_status()
                current_session = mkt["session_name"]

                if self._last_market_session != current_session:
                    if current_session == "REGULAR_MARKET_OPEN":
                        self.state_tracker._reset_daily_circuit_breaker()
                        self.ws_streamer.reset_session_ticks()
                        
                        system_logger.log("TRADE", "MarketSession", f"??? ?? ? ? ({mkt['now_kst_str']})")
                        open_msg = f"""🏁 <b>[정규장 매매 개시]</b>\n\n🕒 현재 시각: `{mkt['now_kst_str']}`\n🤖 실행 모드: `{self.broker.mode_str}`\n🧠 AI 전략: `하이브리드 MoE V3 (09:30~15:30 EDT)`\n🎯 매수 룰: `GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.0f}% 이상`\n🛡 방어 룰: `Max TP +{config.MAX_TP_PCT*100:.1f}% / {config.TRAILING_TRIGGER_PCT*100:.1f}% 도달 시 발동, -{config.TRAILING_DROP_PCT*100:.1f}% 하락 시 익절`\n⏰ 마감 룰: `15:50 EDT 0% 오버나잇 전량 시장가`"""
                        self.dispatcher.send_telegram_message(open_msg)

                    elif current_session in ["AFTER_MARKET_CLOSED", "CLOSED"]:
                        eod_date_file = DATA_DIR / "last_eod_date.json"
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        already_run = False
                        if eod_date_file.exists():
                            try:
                                with open(eod_date_file, "r", encoding="utf-8") as f:
                                    d = json.load(f)
                                if d.get("eod_date") == today_str:
                                    already_run = True
                            except Exception:
                                pass

                        if not already_run:
                            try:
                                from core.data_lake import DailyAutoPipeline
                                pipeline = DailyAutoPipeline(self.data_lake)
                                eod_res = pipeline.run_full_eod_pipeline(send_telegram=True)
                                with open(eod_date_file, "w", encoding="utf-8") as f:
                                    json.dump({"eod_date": today_str, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                            except Exception as e:
                                system_logger.info(f"[EOD Pipeline Error] {e}")

                    self._last_market_session = current_session

                # === AI Evaluation & Briefing ===
                if current_session == "REGULAR_MARKET_OPEN" and not self._is_order_in_progress:
                    now_t = time.time()
                    active_pos = self.state_tracker._get_active_position()
                    
                    now_dt = datetime.now()
                    force_first_run = (self._last_briefing_time == 0.0)
                    is_target_minute = (now_dt.minute % 15 == 0)
                    
                    if (is_target_minute or force_first_run) and (now_t - self._last_briefing_time) >= 60:
                        self._last_briefing_time = now_t
                        
                        try:
                            self.data_lake.sync_live_intraday_candles([config.TICKER_LONG, config.TICKER_SHORT, config.TICKER_TREND, config.MACRO_TICKER_2, config.MACRO_TICKER_1, config.MACRO_TICKER_3, config.MACRO_TICKER_4])
                            df_15m = self.data_lake.load_candles(config.TICKER_LONG, "15m")
                            live_prices = {
                                config.TICKER_LONG: self.ws_streamer.get_latest_price(config.TICKER_LONG, 0.0),
                                config.TICKER_SHORT: self.ws_streamer.get_latest_price(config.TICKER_SHORT, 0.0),
                                config.TICKER_TREND: self.ws_streamer.get_latest_price(config.TICKER_TREND, 0.0),
                                config.MACRO_TICKER_1: self.ws_streamer.get_latest_price(config.MACRO_TICKER_1, 0.0),
                                config.MACRO_TICKER_2: self.ws_streamer.get_latest_price(config.MACRO_TICKER_2, 0.0),
                                config.MACRO_TICKER_3: self.ws_streamer.get_latest_price(config.MACRO_TICKER_3, 0.0),
                                config.MACRO_TICKER_4: self.ws_streamer.get_latest_price(config.MACRO_TICKER_4, 0.0),
                            }
                            
                            moe_res = self.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                            self._last_moe_res = moe_res
                            
                            conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                            is_appr = moe_res.get('is_approved', False)
                            direction = moe_res.get('direction', 'NONE')
                            gbdt_probs = moe_res.get('gbdt_probs', {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34})
                            self.briefing_manager.send_ai_briefing(now_dt, moe_res, direction, is_appr, active_pos, self.state_tracker.daily_circuit_breaker_triggered, gbdt_probs)
                            
                            if is_appr and not active_pos and not self.state_tracker.daily_circuit_breaker_triggered:
                                if direction in [f"LONG_{config.TICKER_LONG}", f"SHORT_{config.TICKER_SHORT}"]:
                                    winner_sym = config.TICKER_LONG if direction == f"LONG_{config.TICKER_LONG}" else config.TICKER_SHORT
                                    cur_px = live_prices.get(winner_sym, 0.0)
                                    
                                    if cur_px > 0:
                                        conn_res = self.broker.test_connection()
                                        usd_avail = float(conn_res.get("usd_order_available", 0.0))
                                        order_qty = int((usd_avail * config.MAX_ALLOCATION_RATIO) / (cur_px + config.QTY_CALC_BUFFER))
                                        
                                        if order_qty > 0:
                                            system_logger.log("TRADE", "AI", f"V3 AI  ? ?! {winner_sym} {order_qty}? ?")
                                            atr_14 = float(moe_res.get('features', {}).get('ATR_14', cur_px * 0.018))
                                            atr_pct = (atr_14 / cur_px) * 100.0 if cur_px > 0 else 1.8
                                            sl_pct = config.SL_MIN_PCT * 100.0
                                            tp_pct = config.MAX_TP_PCT * 100.0
                                            
                                            targets = {
                                                "dynamic_tp_px": round(cur_px * (1.0 + tp_pct/100.0), 2),
                                                "dynamic_sl_px": round(cur_px * (1.0 - sl_pct/100.0), 2),
                                                "tp_pct": tp_pct,
                                                "sl_pct": sl_pct,
                                                "time_stop_minutes": 90,
                                                "strategy_tag": "Lumos V4 Optimal"
                                            }
                                            self.order_engine._execute_buy_with_10s_chase(
                                                symbol=winner_sym,
                                                target_qty=order_qty,
                                                ref_price=cur_px,
                                                moe_res=moe_res,
                                                targets=targets
                                            )
                        except Exception as e:
                            system_logger.log("ERROR", "AI", f"AI ? ?? : {e}")


                if mkt["is_open"] and not self._is_order_in_progress:
                    try:
                        stk_bal = self.broker.get_overseas_stock_balance()
                        if stk_bal.get("ok") and stk_bal.get("holdings"):
                            live_p = {
                                config.TICKER_LONG: self.ws_streamer.get_latest_price(config.TICKER_LONG, 0.0),
                                config.TICKER_SHORT: self.ws_streamer.get_latest_price(config.TICKER_SHORT, 0.0)
                            }
                            if current_session == "EOD_LIQUIDATION":
                                for h in stk_bal.get("holdings"):
                                    sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
                                    qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                                    if qty > 0:
                                        system_logger.log("TRADE", "EOD", f"Executing 15:50 EOD Liquidation for {sym}")
                                        self.order_engine._execute_sell_with_10s_chase(
                                            symbol=sym,
                                            quantity=qty,
                                            reason_desc="EOD 15:50 100% Liquidation",
                                            is_market_order=True
                                        )
                            else:
                                self._manage_open_positions(stk_bal, realtime_px_override=live_p)
                    except Exception as e:
                        pass
                sleep_seconds = 3 if mkt["is_open"] else 15
                time.sleep(sleep_seconds)

            except Exception as e:
                system_logger.log("ERROR", "LiveRunner", f" ? : {e}")
                time.sleep(1)

    def run_once_and_start_listener(self):
        """
        1???? ? ??????  ????? ??? ??
        """
        system_logger.info("=" * 75)
        # 0. ? [?  ??? ??1?  ? ???&  ??? ?
        system_logger.info("\n" + "=" * 75)
        system_logger.info("?  [?  ??? ??  ? ???&  ??? ?  ?")
        system_logger.info("=" * 75)
        sync_res = KRMarketCalendar.verify_time_synchronization()
        for chk_item in sync_res["checklist_items"]:
            system_logger.info(f"   ??{chk_item}")
        if not sync_res["all_ok"]:
            err_msg = "? [??? ????  ???? ??????? ??? ??!"
            system_logger.critical(err_msg)
            raise RuntimeError(err_msg)
        system_logger.info(f"   ??? ? ????100% ??? ({sync_res['dst_text']})")
        system_logger.info("=" * 75)

        # 1.  ? ? ?
        mkt_status = KRMarketCalendar.get_market_status()
        self._last_market_session = mkt_status["session_name"]

        system_logger.info(f"\n[1/3] ?  ? ? ??:")
        system_logger.info(f"   ??? ?: {mkt_status['status_desc']}")
        system_logger.info(f"   ??? ?: {mkt_status['now_kst_str']}")
        system_logger.info(f"   ??? ?: {mkt_status['now_kr_str']} ({mkt_status['dst_text']})")
        system_logger.info(f"   ??? : {mkt_status['next_open_kst_str']} (?? ?: {mkt_status['time_until_open_str']})")

        # 2. ? ?? ? ??
        system_logger.info(f"\n[2/3] {self.broker.broker_name} {self.broker.mode_str}  ?? ? ? ??:")
        conn_res = self.broker.test_connection()
        system_logger.info(f"   ??OAuth2 ?: {'???  ?' if conn_res['ok'] else '?? ?'}")
        system_logger.info(f"   ??? : {self.broker.account_no}-{self.broker.account_type}")
        system_logger.info(f"   ?????: ${conn_res['usd_order_available']:,.2f} USD")

        # 3. ?  ? ??????
        ai_engine_desc = f"`Phase 1: 15m GBDT Model ({config.GBDT_CONFIDENCE_THRESHOLD*100:.0f}% 이상 진입 / 크로스에셋 Veto 방패 / GBDT Feature 융합)`"
        start_msg = f"""🏁 <b>[Lumos 라이브러너 기동] {self.broker.mode_str} 자동매매 프로세스 시작</b>
🕒 <b>현재 시각:</b> `{mkt_status['now_kst_str']}`
✅ <b>시계 동기화:</b> `100% 일치 (KST-KST {sync_res['delta_hours']:.0f}h 시차 검증 완료)`
📊 <b>시장 상태:</b> `{mkt_status['status_desc']}`
⏰ <b>정규장 개장:</b> `{mkt_status['next_open_kst_str']}` ({mkt_status['time_until_open_str']})
🏦 <b>증권사 연동:</b> `{self.broker.broker_name} ({self.broker.mode_str})`
💼 <b>계좌 번호:</b> `{self.broker.account_no}-{self.broker.account_type}`
💰 <b>가용 예수금:</b> `${conn_res['usd_order_available']:,.2f} USD` (약 `{conn_res['krw_converted']:,}원`)
📦 <b>보유 종목수:</b> `{conn_res['holdings_count']}개`
🤖 <b>AI 코어엔진:</b> {ai_engine_desc}
⚡ <b>데이터 피드:</b> `10ms WebSocket 초고속 스트리밍 구독 완료`
준비 완료, 정규장 개장까지 틱 데이터를 백그라운드 수집합니다."""
        try:
            self.dispatcher.send_telegram_message(start_msg)
            system_logger.log("INFO", "LiveRunner", f"? {self.broker.mode_str} ?  ? ??? (: {self.broker.account_no}, ?? ${conn_res['usd_order_available']:,.2f})")
        except Exception as te:
            system_logger.warn(f"? ? ? ?: {te}")
        
        # 4. ????WebSocket) ??? ??
        self.ws_streamer.start()
        system_logger.info(f"??[{self.broker.broker_name} ??WebSocket ? ???] 10ms ??????? ?")

        # 5. ????   ?????
        trd_thread = threading.Thread(target=self._market_execution_loop, daemon=True)
        trd_thread.start()

        # 6. ??? ????? ? (Foreground Daemon)
        self.controller.listen_loop(poll_interval=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lumos Auto Trading Live Runner")
    parser.add_argument("--real", action="store_true", help="Run with real money broker")
    parser.add_argument("--sim", action="store_true", help="Run with simulation/mock broker")
    args, unknown = parser.parse_known_args()

    is_sim = None
    if getattr(args, 'real', False):
        is_sim = False
    elif getattr(args, 'sim', False):
        is_sim = True

    runner = KiwoomLiveRunner(is_simulation=is_sim)
    runner.run_once_and_start_listener()

