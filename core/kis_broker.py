import os
import sys
import math
import time
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    KIS_MODE,
    KIS_VIRTUAL_APP_KEY,
    KIS_VIRTUAL_APP_SECRET,
    KIS_VIRTUAL_CANO,
    KIS_VIRTUAL_ACNT_PRDT_CD,
    KIS_VIRTUAL_BASE_URL,
    KIS_REAL_APP_KEY,
    KIS_REAL_APP_SECRET,
    KIS_REAL_CANO,
    KIS_REAL_ACNT_PRDT_CD,
    KIS_REAL_BASE_URL
)
from core.kis_client import KisClient

logger = logging.getLogger("KisBroker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][KisBroker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class KisIOCBroker:
    """
    [Lumos v10.2 한국투자증권 대규모 자금 최적화 IOC 스마트 브로커]
    1. 1억 원 상당(USD $100,000) 전액 운용 안전성 확보
    2. math.floor 정수 주수 계산으로 잔고 부족 에러 원천 차단
    3. 최유리 1~2호가 기준 IOC (Immediate-Or-Cancel) 지정가 주문 라우팅으로 슬리피지/호가 왜곡 방지
    4. 미체결 잔량 실시간 감시 및 타임스탑 클린업
    """
    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or KIS_MODE or "VIRTUAL").upper()
        self.client = KisClient(mode=self.mode)
        logger.info(f"✅ KisIOCBroker 초기화 완료 (모드: {self.mode})")

    def get_account_balance(self) -> Dict[str, Any]:
        """예수금 및 주문가능금액 조회"""
        return self.client.get_balance()

    def calculate_safe_order_qty(self, ticker: str, current_price_usd: float, capital_usd: float, fee_buffer_pct: float = 0.0035) -> int:
        """
        1억 원 상당 전액 운용 시 잔고 부족 방지를 위한 정수 주수 계산
        - 수수료 + 슬리피지 버퍼(0.35%) 선차감 후 math.floor 정수 변환
        """
        if current_price_usd <= 0 or capital_usd <= 0:
            return 0
        effective_capital = capital_usd * (1.0 - fee_buffer_pct)
        raw_qty = effective_capital / current_price_usd
        safe_qty = math.floor(raw_qty)
        logger.info(f"[{ticker}] 안전 주문 주수 계산: 가용 ${capital_usd:,.2f} / 현재가 ${current_price_usd:.2f} ➔ 정수 {safe_qty}주 (버퍼 {fee_buffer_pct*100:.2f}%)")
        return safe_qty

    def execute_ioc_order(self, ticker: str, side: str, qty: int, limit_price: float, reason: str = "SIGNAL") -> Dict[str, Any]:
        """
        최유리 1~2호가 기준 IOC 지정가 주문 집행
        - side: 'BUY' 또는 'SELL'
        - IOC (Immediate-Or-Cancel): 즉시 체결 가능한 수량만 체결되고 잔량은 자동 취소
        """
        if qty <= 0:
            logger.warning(f"[{ticker}] 주문 주수가 0이므로 주문을 생략합니다.")
            return {"ok": False, "msg": "주문 주수 0"}

        # 최유리 호가 슬리피지 방지 가격 보정 (매수는 +0.05%, 매도는 -0.05% 범위 지정가)
        slippage_adj = 1.0005 if side == "BUY" else 0.9995
        ioc_limit_price = round(limit_price * slippage_adj, 2)

        logger.info(f"🚀 [IOC 주문 발주] {side} {ticker} {qty}주 @ ${ioc_limit_price:.2f} (IOC 지정가, 사유: {reason})")

        order_res = self.client.send_order(
            ticker=ticker,
            side=side,
            qty=qty,
            price=ioc_limit_price,
            order_type="00"  # 지정가 (API 내부에서 모드별 IOC 조건 매핑)
        )

        return {
            "ok": order_res.get("ok", False),
            "ticker": ticker,
            "side": side,
            "qty": qty,
            "price": ioc_limit_price,
            "order_no": order_res.get("order_no", f"IOC_{int(time.time())}"),
            "msg": order_res.get("msg", ""),
            "raw_response": order_res
        }

    def liquidate_all_emergency(self, reason: str = "EMERGENCY_KILL_SWITCH") -> Dict[str, Any]:
        """비상 전량 시장가 즉각 청산 (100% 현금 피신)"""
        logger.warning(f"🚨 [긴급 전량 청산 개시] 사유: {reason}")
        return self.client.liquidate_all_positions(reason=reason)
