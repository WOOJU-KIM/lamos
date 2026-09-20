import time
import math
import config
from typing import Dict, Any, Optional, Tuple
from core.system_logger import system_logger
from core.state_tracker import StateTracker

class OrderExecutionEngine:
    def __init__(self, broker, dispatcher, state_tracker: StateTracker, ws_streamer=None):
        self.broker = broker
        self.dispatcher = dispatcher
        self.state_tracker = state_tracker
        self.ws_streamer = ws_streamer
        import threading
        self._order_lock = threading.Lock()
        self.order_timeout_buy_sec = 3.0
        self.order_timeout_sell_sec = 3.0

    def _wait_for_fill(self, order_no: str, symbol: str, is_buy: bool, target_qty: int, timeout_sec: Optional[float] = None) -> Tuple[bool, int, int]:
        """
        """
        if timeout_sec is None:
            timeout_sec = self.order_timeout_buy_sec if is_buy else self.order_timeout_sell_sec

        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            ws_filled = self._order_fills.get(order_no, 0)
            if ws_filled >= target_qty:
                return True, ws_filled, 0
            time.sleep(config.ORDER_POLL_INTERVAL_SEC / 10.0)

        # ????? ????? 1?? ?
        stk_bal = self.broker.get_overseas_stock_balance()
        actual_qty = 0
        if stk_bal.get("ok"):
            for h in stk_bal.get("holdings", []):
                if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                    actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                    break

        if is_buy:
            unfilled = max(0, target_qty - actual_qty)
            return (actual_qty >= target_qty), actual_qty, unfilled
        else:
            return (actual_qty <= 0), target_qty - actual_qty, actual_qty

    def _execute_buy_with_10s_chase(
        self,
        symbol: str,
        target_qty: int,
        ref_price: float,
        moe_res: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> bool:
        """
        """
        with self._order_lock:
            self._is_order_in_progress = True

        try:
            if target_qty <= 0:
                system_logger.warn(f"⚠️ [매수 거부] 비정상 타겟 수량({target_qty}주)으로 진입이 취소되었습니다.")
                return False

            exp_name = moe_res.get("expert_desc", "MoE Gating")
            top_conf = float(moe_res.get("gating_confidence", 0.0)) * 100.0
            tp_px = targets["dynamic_tp_px"]
            sl_px = targets["dynamic_sl_px"]
            tp_pct = targets.get("tp_pct", 3.0)
            sl_pct = targets.get("sl_pct", 2.0)
            time_stop_min = targets.get("time_stop_minutes", 90)
            strategy_tag = targets.get("strategy_tag", "Lumos V3")
            atr_14 = targets.get("atr_14", 0.0)
            all_scores = moe_res.get("all_gating_confidences", {})

            # ----------------------------------------------------
            # 1. 1?  (Ask + $0.03)
            # ----------------------------------------------------
            cur_px = float(self.ws_streamer.get_latest_price(symbol, ref_price))
            if cur_px <= 0:
                cur_px = ref_price
            order_px_1 = round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)

            system_logger.info(f"?? [1? ] {symbol} {target_qty}?@ ${order_px_1:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            ord_res_1 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=target_qty, price=order_px_1)
            ord_no_1 = str(ord_res_1.get("order_no", "")).strip()

            # ?  ?  (1? )
            mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
            score_block = "\n".join([f"??**{k.upper()}:** `{v*100:.1f}??" for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)])
            buy_msg = f"""{mode_title} 신규 매수 주문 (1차)\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\n💡 **종목:** `{symbol}`\n📊 **수량:** `{target_qty}`\n💰 **주문가:** `${order_px_1:.2f}`\n\n🔥 **[GBDT 모델 확신도]**\n{score_block}\n\n🎯 **목표가:** `${tp_px:.2f}`\n🛡️ **손절가:** `${sl_px:.2f}`\n⏱️ **시간청산:** {time_stop_min}분"""


            self.dispatcher.send_telegram_message(buy_msg)
            system_logger.log("TRADE", "AutoExecution", f"🚀 [{self.broker.broker_name}] {symbol} 1차 매수 진입 - {target_qty}주 @ ${order_px_1:.2f} (AI확신도: {top_conf:.1f}%)")

            # 1. ???(?  ??0ms   ?)
            is_filled_1, filled_1, unfilled_1 = self._wait_for_fill(
                order_no=ord_no_1,
                symbol=symbol,
                is_buy=True,
                target_qty=target_qty,
                timeout_sec=self.order_timeout_buy_sec
            )

            if is_filled_1:
                # 1??100%  ?! (??? ? ?? 100% ???
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_1, target_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 1
                }
                self.state_tracker._save_active_position(filled_pos)
                self.experience_system_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=1, order_no=ord_no_1,
                    price=real_buy_px, quantity=real_qty, status="FILLED",
                    note=f"1? 100%  ? (? ? : ${real_buy_px:.2f})"
                )
                system_logger.info(f"??[1? 100%  ?] {symbol} {real_qty}?@ ${real_buy_px:.2f} (? ?: {filled_pos['buy_time']})")
                self.notifier.send_entry_alert(
                    ticker=symbol,
                    entry_price=real_buy_px,
                    qty=real_qty,
                    gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                    cross_dir=moe_res.get("cross_dir", "HOLD"),
                    tp_price=tp_px_dyn,
                    sl_price=sl_px_dyn,
                    time_stop_minutes=time_stop_min,
                    strategy_tag=strategy_tag,
                    is_simulation=self.broker.is_simulation
                )
                return True

            # ----------------------------------------------------
            # 2. 1?? & 2???(??1??)
            # ----------------------------------------------------
            system_logger.info(f"??[1? {self.order_timeout_buy_sec:.0f}??({unfilled_1}?] 1?  ?? ???1????")
            self.experience_system_logger.record_order_event(
                symbol=symbol, action="BUY", attempt=1, order_no=ord_no_1,
                price=order_px_1, quantity=unfilled_1, status="UNFILLED_TIMEOUT",
                note=f"1? {self.order_timeout_buy_sec:.0f}?? ??2???"
            )
            if ord_no_1:
                self.broker.cancel_order(order_no=ord_no_1, symbol=symbol, quantity=unfilled_1)
            time.sleep(config.ORDER_POLL_INTERVAL_SEC)

            # ? ? ?? ???? 
            stk_bal_1 = self.broker.get_overseas_stock_balance(force_refresh=True)
            already_filled_qty = 0
            if stk_bal_1.get("ok"):
                for h in stk_bal_1.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        already_filled_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break

            remaining_qty = target_qty - already_filled_qty
            if remaining_qty <= 0:
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_1, already_filled_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 1
                }
                self.state_tracker._save_active_position(filled_pos)
                self.notifier.send_entry_alert(
                    ticker=symbol,
                    entry_price=real_buy_px,
                    qty=real_qty,
                    gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                    cross_dir=moe_res.get("cross_dir", "HOLD"),
                    tp_price=tp_px_dyn,
                    sl_price=sl_px_dyn,
                    time_stop_minutes=time_stop_min,
                    strategy_tag=strategy_tag,
                    is_simulation=self.broker.is_simulation
                )
                return True

            # 2??? ???
            latest_px_2 = float(self.ws_streamer.get_latest_price(symbol, cur_px))
            order_px_2 = round(latest_px_2 + config.BUY_SLIPPAGE_ADJUST, 2)
            system_logger.info(f"??[2? ??(1/1)] {symbol} {remaining_qty}?@ ${order_px_2:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            ord_res_2 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=remaining_qty, price=order_px_2)
            ord_no_2 = str(ord_res_2.get("order_no", "")).strip()
            system_logger.log("TRADE", "OrderChasing", f"⚠️ [1차 미체결] {symbol} 2차 추격 매수 발주 - {remaining_qty}주 @ ${order_px_2:.2f}")

            # 2. ???(?  ??0ms   ?)
            is_filled_2, filled_2, unfilled_2 = self._wait_for_fill(
                order_no=ord_no_2,
                symbol=symbol,
                is_buy=True,
                target_qty=remaining_qty,
                timeout_sec=self.order_timeout_buy_sec
            )

            if is_filled_2:
                # 2??100%  ?! (??? ? ?? 100% ???
                total_qty = already_filled_qty + remaining_qty
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_2, total_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 2
                }
                self.state_tracker._save_active_position(filled_pos)
                self.experience_system_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=2, order_no=ord_no_2,
                    price=real_buy_px, quantity=remaining_qty, status="FILLED",
                    note=f"2? 100%  ? (? ? : ${real_buy_px:.2f})"
                )
                system_logger.info(f"??[2? 100%  ?] {symbol} {real_qty}?@ ${real_buy_px:.2f} (? ?: {filled_pos['buy_time']})")
                self.notifier.send_entry_alert(
                    ticker=symbol,
                    entry_price=real_buy_px,
                    qty=real_qty,
                    gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                    cross_dir=moe_res.get("cross_dir", "HOLD"),
                    tp_price=tp_px_dyn,
                    sl_price=sl_px_dyn,
                    time_stop_minutes=time_stop_min,
                    strategy_tag=strategy_tag,
                    is_simulation=self.broker.is_simulation
                )
                return True

            # ----------------------------------------------------
            # 3. 2???????  ????100% 
            # ----------------------------------------------------
            system_logger.info(f"? [2? 10??({unfilled_2}?]  ?  ????100%  ??AI ???????")
            self.experience_system_logger.record_order_event(
                symbol=symbol, action="BUY", attempt=2, order_no=ord_no_2,
                price=order_px_2, quantity=unfilled_2, status="CASH_PRESERVED",
                note="2? ??  (??100%  ??AI ???????)"
            )
            if ord_no_2:
                self.broker.cancel_order(order_no=ord_no_2, symbol=symbol, quantity=unfilled_2)
            time.sleep(config.ORDER_POLL_INTERVAL_SEC)

            # ? ???  ?
            stk_bal_final = self.broker.get_overseas_stock_balance(force_refresh=True)
            final_filled = 0
            if stk_bal_final.get("ok"):
                for h in stk_bal_final.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        final_filled = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break

            if final_filled > 0:
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_2, final_filled)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 2
                }
                self.state_tracker._save_active_position(filled_pos)
                system_logger.info(f"? [?  ?] {symbol} {real_qty}??? (? ??: ${real_buy_px:.2f})")
                self.notifier.send_entry_alert(
                    ticker=symbol,
                    entry_price=real_buy_px,
                    qty=real_qty,
                    gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                    cross_dir=moe_res.get("cross_dir", "HOLD"),
                    tp_price=tp_px_dyn,
                    sl_price=sl_px_dyn,
                    time_stop_minutes=time_stop_min,
                    strategy_tag=strategy_tag,
                    is_simulation=self.broker.is_simulation
                )
                return True
            else:
                self.state_tracker._clear_active_position()
                system_logger.log("TRADE", "OrderChasing", f"🛑 [매수 취소] {symbol} 2차 추격 미체결로 매수 포기 (예수금 100% 보존, 스마트 주문취소)")
                return False

        except Exception as e:
            system_logger.error(f"??   ?: {e}")
            system_logger.log("ERROR", "BuyChase", f"?   ? : {e}")
            try:
                self.experience_system_logger.record_error_event("LiveRunner", "BUY_CHASE_EXCEPTION", "ERROR", str(e))
            except Exception:
                pass
            return False
        finally:
            with self._order_lock:
                self._is_order_in_progress = False
                system_logger.info("? [AI ???? ? ?] ? ? ????")

    def _execute_sell_with_10s_chase(
        self,
        symbol: str,
        quantity: int,
        reason_desc: str,
        buy_px: float = 0.0,
        cur_px: float = 0.0,
        is_stoploss: bool = False,
        is_market_order: bool = False
    ) -> bool:
        """
        """
        with self._order_lock:
            self._is_order_in_progress = True

        try:
            b_name = self.broker.broker_name
            loop_retry = 0

            # ? [??? ? ? (Hard Rule #2)]
            # ? ?? ??? ????(poss_qty) ?
            stk_bal_check = self.broker.get_overseas_stock_balance()
            actual_qty = 0
            if stk_bal_check.get("ok"):
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
            if actual_qty <= 0:
                system_logger.info(f"? [? ? 0??] {symbol} ?? 100% ? ? ? ?? ? ?AI ???? ?")
                self.state_tracker._clear_active_position()
                return True
            quantity = min(quantity, actual_qty)

            while True:
                loop_retry += 1
                latest_px = float(self.ws_streamer.get_latest_price(symbol, cur_px))
                if latest_px <= 0:
                    latest_px = cur_px

                if is_market_order and not self.broker.is_simulation:
                    # 15:50 EOD ?? 0%  ?: ?? ?? ?(03, price=0.0)  
                    sell_px = 0.0
                    order_label = "? ? ?(Market Order)"
                else:
                    # ? ? ? (? ?? ? ?? ? ): Bid - $0.05 ????
                    sell_px = max(0.01, round(latest_px - config.SELL_SLIPPAGE_ADJUST, 2)) if latest_px > 0 else 0.0
                    order_label = f"?  ??(${sell_px:.2f})"

                system_logger.info(f"? [ ?  ({loop_retry}?)] {symbol} {quantity}?| {order_label} | ?: {reason_desc} ({self.order_timeout_sell_sec:.0f}? ??")
                s_res = self.broker.send_order(symbol=symbol, order_type="SELL", quantity=quantity, price=sell_px)
                s_ord_no = str(s_res.get("order_no", "")).strip()

                # ? 3? ??(?  ??0ms   ?)
                is_sold, sold_qty, rem_qty = self._wait_for_fill(
                    order_no=s_ord_no,
                    symbol=symbol,
                    is_buy=False,
                    target_qty=quantity,
                    timeout_sec=self.order_timeout_sell_sec
                )

                if is_sold:
                    # 100% ? ? ?!
                    active_pos = self._get_active_position() or {}
                    self.state_tracker._clear_active_position()
                    final_pnl_pct = ((latest_px - buy_px) / buy_px * 100) if buy_px > 0 else 0.0

                    # ? [AI ?? ?? ???Dual CSV+DB) ? ?]
                    high_px = active_pos.get("high_price_during_hold", max(buy_px, latest_px))
                    low_px = active_pos.get("low_price_during_hold", min(buy_px, latest_px))
                    mfe_pct = round(((high_px - buy_px) / buy_px) * 100, 2) if buy_px > 0 else 0.0
                    mae_pct = round(((low_px - buy_px) / buy_px) * 100, 2) if buy_px > 0 else 0.0
                    try:
                        self.experience_system_logger.record_trade({
                            "mode": self.broker.mode_str,
                            "symbol": symbol,
                            "entry_time": active_pos.get("buy_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "intended_entry_price": active_pos.get("ref_price", buy_px),
                            "actual_entry_price": buy_px,
                            "intended_exit_price": sell_px if sell_px > 0 else latest_px,
                            "actual_exit_price": latest_px,
                            "quantity": quantity,
                            "pnl_pct": final_pnl_pct,
                            "pnl_usd": round((latest_px - buy_px) * quantity, 2),
                            "exit_reason": reason_desc,
                            "mfe_pct": mfe_pct,
                            "mae_pct": mae_pct,
                            "gbdt_confidence": active_pos.get("gbdt_confidence", 0.0),
                            "cross_dir": active_pos.get("cross_dir", "HOLD"),
                            "buy_attempts": active_pos.get("buy_attempts", 1),
                            "sell_attempts": loop_retry,
                            "features": active_pos.get("features", {})
                        })
                        self.experience_system_logger.record_order_event(
                            symbol=symbol, action="SELL", attempt=loop_retry, order_no=s_ord_no,
                            price=sell_px, quantity=quantity, status="FILLED", note=f"? ?: {reason_desc}"
                        )
                    except Exception as le:
                        system_logger.info(f"  ? ? (): {le}")

                    mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
                    system_logger.log("TRADE", "Liquidation", f"🎯 [전량 청산 완료] {symbol} {quantity}주 | 사유: {reason_desc} | 최종 수익률: {final_pnl_pct:+.2f}%")

                    sell_msg = f"""{mode_title} 100% 매도 청산 완료\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\n💡 **사유:** `{reason_desc}`\n📊 **종목/수량:** `{symbol} {quantity:,}주`\n💰 **체결가:** `${latest_px:.2f}` (최종 수익률: {final_pnl_pct:+.2f}%)\n🛡️ **사후 관리:** `100% 현금화 완료 (AI MoE 새 진입 대기)`"""
                    self.dispatcher.send_telegram_message(sell_msg)
                    if is_stoploss:
                        self.state_tracker._record_stoploss()
                    return True

                if s_ord_no:
                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)
                    time.sleep(config.ORDER_POLL_INTERVAL_SEC)

        except Exception as e:
            system_logger.error(f"매도 추격 주문 에러: {e}")
            system_logger.log("ERROR", "SellChase", f"매도 추격 에러: {e}")
            try:
                self.experience_system_logger.record_error_event("LiveRunner", "SELL_CHASE_EXCEPTION", "ERROR", str(e))
            except Exception:
                pass
            return False
        finally:
            with self._order_lock:
                self._is_order_in_progress = False
                system_logger.info("🔒 [AI 포지션 청산 완료] 주문 락 해제완료")
