from typing import Dict, Any, Optional
from datetime import datetime
from core.system_logger import system_logger
import config

class BriefingManager:
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        
    def send_ai_briefing(self, now_dt: datetime, moe_res: Dict[str, Any], direction: str, is_appr: bool, active_pos: Optional[Dict[str, Any]], circuit_breaker_triggered: bool, gbdt_probs: Dict[str, Any]):
        p_long = float(gbdt_probs.get('LONG', 0.0)) * 100.0
        p_short = float(gbdt_probs.get('SHORT', 0.0)) * 100.0
        p_none = float(gbdt_probs.get('NONE', 0.0)) * 100.0
        conf = float(moe_res.get('confidence', 0.0)) * 100.0
        threshold = float(moe_res.get('threshold_applied', 0.6)) * 100.0
        
        dir_str = "매수 관망" if direction == "NONE" else direction
        briefing_msg = f"""🤖 <b>[Lumos AI 정기 브리핑]</b>
⏰ <b>시간</b>: {now_dt.strftime('%H:%M')} (KST)
🧭 <b>AI 판독 방향</b>: {dir_str}
🎯 <b>진입 임계값</b>: {threshold:.1f}%
📈 <b>롱({config.TICKER_LONG}) 확률</b>: {p_long:.1f}%
📉 <b>숏({config.TICKER_SHORT}) 확률</b>: {p_short:.1f}%
⏸ <b>관망 확률</b>: {p_none:.1f}%
🔥 <b>최종 GBDT 확신도</b>: {conf:.1f}%
🚦 <b>상태</b>: {'진입 승인 🟢' if is_appr else '관망 유지 🟡'}
"""
        if self.dispatcher:
            self.dispatcher.send_telegram_message(briefing_msg)
            
        detailed_log = f"15분봉 AI 브리핑 [방향: {dir_str}, 롱 {p_long:.1f}%, 숏 {p_short:.1f}%, 관망 {p_none:.1f}%, 상태: {'진입 승인' if is_appr else '관망'}]"
        system_logger.info(detailed_log, source="AI")
