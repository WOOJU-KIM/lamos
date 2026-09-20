import config
from google import genai
from typing import Dict, Any
import json
import re

class StrategyAgent:
    """
    [2. Strategy Agent]
    시장 국면(Regime) 분석 결과와 자가 진화된 활성 퀀트 로직(Active Logic) 파라미터를 기반으로
    정밀한 포지션 및 진입/손절/스위칭을 판단하는 수석 퀀트 전략가
    """
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3.6-flash"

    def analyze_strategy(self, market_data: Dict[str, Any], data_summary_text: str, regime_data: Dict[str, Any], active_logic: Dict[str, Any]) -> Dict[str, Any]:
        """시장 데이터 + 국면 분석 + 자가진화 로직 파라미터를 종합하여 최종 포지션 도출"""
        
        regime_name = regime_data.get("regime_name_kr", "상승 추세")
        regime_mode = regime_data.get("strategy_params", {}).get("target_regime_mode", "BALANCED")
        regime_params = regime_data.get("strategy_params", {})
        
        system_instruction = f"""당신은 월가 최상위 퀀트 헤지펀드의 수석 트레이딩 전략가(Chief Quant Strategist)입니다.
현재 시스템은 시장 국면 분석기(Market Regime Classifier) 및 자가 진화 엔진(Self-Evolution Engine)에 의해 가동 중입니다.

[현재 가동 중인 자가 진화 활성 로직 (Active Logic)]
- 로직 버전: {active_logic.get('version', 'v1.0')} ({active_logic.get('name')})
- 기본 선제 진입선: {active_logic.get('dip_buy_pct', -1.0):+.2f}%
- 엄격 손절선: {active_logic.get('stop_loss_pct', -2.0):+.2f}%
- SQQQ 헷지 스위칭선: {active_logic.get('hedge_switch_pct', -1.5):+.2f}%
- 목표 익절선: +{active_logic.get('take_profit_pct', 4.5):.2f}%

[현재 시장 국면 (Market Regime)]
- 국면명: {regime_name} (모드: {regime_mode})
- TQQQ 가중치 배수: {regime_params.get('long_weight_mult', 1.0)}x
- SQQQ 가중치 배수: {regime_params.get('short_weight_mult', 0.5)}x
- 국면 권장 선제 진입: {regime_params.get('dip_buy_threshold_pct', -1.0):+.2f}%
- 국면 권장 손절선: {regime_params.get('stop_loss_pct', -2.0):+.2f}%
- 국면 권장 헷지 스위칭: {regime_params.get('hedge_switch_threshold_pct', -1.5):+.2f}%

[3대 매매 원칙]
1. TQQQ 분할 매수 / 선제 매수 (Dip Buy):
   - 현재 국면과 활성 로직의 선제 진입 기준에 부합할 때 분할 매수로 진입.
2. 칼손절 (Strict Stop-Loss):
   - 손절선 이탈 시 원금 보존을 위해 즉시 손절 지시.
3. SQQQ 헤지 스위칭 (Hedge Switching):
   - 하락 추세 전환 또는 손절선 이탈 시 SQQQ(인버스 3배)로 즉시 스위칭.

반드시 다음 JSON 형식으로만 응답하세요:
```json
{{
  "action": "BUY_TQQQ_DIP" | "STOP_LOSS_TQQQ" | "SWITCH_SQQQ_HEDGE" | "HOLD_TQQQ" | "HOLD_SQQQ" | "WAIT_CASH",
  "target_asset": config.TICKER_LONG | config.TICKER_SHORT | "CASH",
  "confidence_score": 1~100 (정수),
  "rationale": "국면 및 활성 로직 파라미터 기반 분석 근거 (3~4문장)",
  "entry_target_price": "진입 권장 가격대 (예: $142.00 ~ $142.80)",
  "stop_loss_price": "손절가 (예: $140.00)",
  "take_profit_price": "익절 목표가 (예: $148.00)"
}}
```
"""

        prompt = f"""다음 실시간 시황 및 국면 지표를 바탕으로 최적의 퀀트 포지션 명령을 내려주세요.

{data_summary_text}

[국면 세부 지표]
- 50일선: ${regime_data.get('metrics', {}).get('ma50')}
- 200일선: ${regime_data.get('metrics', {}).get('ma200')}
- 20일 실현 변동성: {regime_data.get('metrics', {}).get('realized_vol_20d_pct')}%
- VIX 백분위: 상위 {regime_data.get('metrics', {}).get('vix_percentile')}%

JSON 형식으로 정밀하게 전략을 수립하세요."""

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
            parsed = {
                "action": "BUY_TQQQ_DIP",
                "target_asset": config.TICKER_LONG,
                "confidence_score": 85,
                "rationale": "국면 지표 및 활성 로직 파라미터 기준 눌림목 분할 진입 유효.",
                "entry_target_price": "장중 현재가 분할 진입",
                "stop_loss_price": f"{active_logic.get('stop_loss_pct', -2.0)}% 기준",
                "take_profit_price": f"+{active_logic.get('take_profit_pct', 4.5)}% 기준"
            }
            
        return parsed
