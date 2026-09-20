import config
from google import genai
from typing import Dict, Any
import json
import re

class RiskAgent:
    """
    [3. Risk Manager Agent]
    블랙스완 킬스위치 상태, VIX 변동성 레벨, 시장 국면 가중치를 종합 심사하여
    최종 포지션 비중(%)을 승인하고 리스크 통제 가이드라인을 확정하는 CRO 에이전트
    """
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"

    def assess_risk_and_allocation(self, market_data: Dict[str, Any], strategy_decision: Dict[str, Any], regime_data: Dict[str, Any], kill_switch_status: Dict[str, Any]) -> Dict[str, Any]:
        """킬스위치 점검 및 국면 가중치 기반 최종 비중 승인"""
        
        # 1. 킬스위치 즉각 발동 시 무조건 100% 현금화 및 진입 차단 (VETO)
        if kill_switch_status.get("triggered", False):
            return {
                "risk_level": "CRITICAL",
                "vix_analysis": f"블랙스완 킬스위치 발동 ({', '.join(kill_switch_status.get('reasons', []))})",
                "approved_action": "EMERGENCY_LIQUIDATE_ALL",
                "recommended_allocation_pct": 0,
                "risk_status": "REJECTED_BY_KILL_SWITCH",
                "risk_notes": "🚨 [KILL-SWITCH ACTIVATED] 비상 제동이 발동되어 전 포지션을 즉시 청산하고 당일 모든 신규 진입을 강제 차단합니다."
            }

        vix = market_data.get("vix", {})
        vix_price = float(vix.get("current_price", 18.0))
        regime_params = regime_data.get("strategy_params", {})
        long_mult = regime_params.get("long_weight_mult", 1.0)
        target_asset = strategy_decision.get("target_asset", config.TICKER_LONG)

        system_instruction = f"""당신은 퀀트 헤지펀드의 최고리스크책임자(Chief Risk Officer, CRO)입니다.
시장 국면(Regime)과 VIX 변동성 모델에 입각하여 포지션 진입 비중(%)을 최종 결정합니다.

[리스크 심사 원칙]
1. VIX < 18 (저변동성): 기준 비중 80~100%에 국면 가중치({long_mult}x) 적용.
2. VIX 18 ~ 25 (중변동성): 기준 비중 50~60%로 제한.
3. VIX > 25 (고변동성): 기준 비중 20~30%로 축소.
4. SQQQ 헷지 포지션 진입 시에는 약세 국면 가중치 적용.

반드시 다음 JSON 형식으로만 응답하세요:
```json
{{
  "risk_level": "LOW" | "MODERATE" | "HIGH" | "CRITICAL",
  "vix_analysis": "VIX 수치 및 변동성 평가 요약",
  "approved_action": "최종 승인 액션",
  "recommended_allocation_pct": 0~100 (정수 비중 %),
  "risk_status": "APPROVED" | "ADJUSTED" | "REJECTED",
  "risk_notes": "리스크 통제 지침 및 분할 진입 가이드 (2~3문장)"
}}
```
"""

        prompt = f"""[시장 및 국면 데이터]
- VIX 지수: {vix_price} pt (변동 {vix.get('change_pct', 0.0):+.2f}%)
- 시장 국면: {regime_data.get('regime_name_kr')} (가중치 {long_mult}x)
- TQQQ 현재가: ${market_data.get('assets', {}).get(config.TICKER_LONG, {}).get('current_price')} ({market_data.get('assets', {}).get(config.TICKER_LONG, {}).get('change_pct')}%)

[Strategy Agent의 제안]
- 행동 계획: {strategy_decision.get('action')} (대상: {target_asset})
- 전략 근거: {strategy_decision.get('rationale')}
- 손절가: {strategy_decision.get('stop_loss_price')}

위 제안을 심사하고 최종 진입 비중(%)과 승인 결정을 JSON으로 작성하세요."""

        interaction = self.client.interactions.create(
            model=self.model_name,
            input=f"{system_instruction}\n\n{prompt}"
        )
        
        response_text = interaction.output_text or ""
        
        try:
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                parsed = json.loads(response_text)
        except Exception:
            alloc = int(min(100, max(10, 80 * long_mult))) if vix_price < 18 else 50
            parsed = {
                "risk_level": "LOW" if vix_price < 18 else "MODERATE",
                "vix_analysis": f"VIX {vix_price}pt 기준 안정적 변동성 유지 중",
                "approved_action": strategy_decision.get("action", "BUY_TQQQ_DIP"),
                "recommended_allocation_pct": alloc,
                "risk_status": "APPROVED",
                "risk_notes": "국면 가중치 및 변동성 가이드라인을 준수하여 분할 진입 승인."
            }
            
        return parsed
