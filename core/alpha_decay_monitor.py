import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple
from config import PORTFOLIO_STATE_FILE, EVOLUTION_LOG_FILE, INITIAL_CAPITAL_KRW

class AlphaDecayMonitor:
    """
    [4. Alpha Decay Monitor & Self-Evolution (로직 수명 감시 및 자동 진화)]
    주간/롤링 성적표 기준 알파 열화(손실 발생, 샤프 저하 등)를 실시간 감지하고,
    Shadow R&D에서 가장 성과가 우수한 상위 후보 로직으로 활성 로직(Active Logic)을 자동 교체/자가진화합니다.
    """
    
    DEFAULT_ACTIVE_LOGIC = {
        "version": "v1.0.0",
        "id": "logic_v1_0_baseline",
        "name": "Base Alpha (원칙 기준)",
        "dip_buy_pct": -1.0,
        "stop_loss_pct": -2.0,
        "hedge_switch_pct": -1.5,
        "take_profit_pct": 4.5,
        "description": "대표님 3대 원칙 기본형",
        "deployed_at": "2026-08-14 00:00:00"
    }

    def __init__(self):
        self.state_file = PORTFOLIO_STATE_FILE
        self.log_file = EVOLUTION_LOG_FILE
        self._ensure_files()

    def _ensure_files(self):
        """상태 파일 및 진화 로그 파일 초기화"""
        if not self.state_file.exists():
            initial_state = {
                "active_logic": self.DEFAULT_ACTIVE_LOGIC,
                "initial_capital_krw": INITIAL_CAPITAL_KRW,
                "current_capital_krw": INITIAL_CAPITAL_KRW,
                "total_pnl_krw": 0,
                "total_return_pct": 0.0,
                "mdd_pct": 0.0,
                "total_trades": 0,
                "win_trades": 0,
                "weekly_returns_history": [0.85, 1.20, -0.45], # 히스토리 샘플
                "last_evolution_at": "2026-08-14 00:00:00",
                "generation": 1
            }
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(initial_state, f, ensure_ascii=False, indent=2)

        if not self.log_file.exists():
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def load_state(self) -> Dict[str, Any]:
        """현재 포트폴리오 및 활성 로직 상태 로드"""
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"active_logic": self.DEFAULT_ACTIVE_LOGIC}

    def save_state(self, state: Dict[str, Any]):
        """상태 저장"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def check_and_evolve(self, shadow_rnd_result: Dict[str, Any], current_week_pnl_pct: float = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        알파 감쇠(Alpha Decay) 검사 및 자가 진화(Self-Evolution) 수행
        Returns: (active_logic, evolution_event_or_none)
        """
        state = self.load_state()
        active_logic = state.get("active_logic", self.DEFAULT_ACTIVE_LOGIC)
        champion = shadow_rnd_result.get("champion_logic", {})
        
        # 최근 주간 수익률 추이 확인
        weekly_history = state.get("weekly_returns_history", [])
        if current_week_pnl_pct is not None:
            weekly_history.append(current_week_pnl_pct)
            state["weekly_returns_history"] = weekly_history[-8:] # 최근 8주 보관
            
        last_week_return = weekly_history[-1] if weekly_history else 0.0
        
        # 알파 열화 판정 조건:
        # 1. 최근 주간 수익률이 음수(-0.1% 이하)이거나
        # 2. Shadow R&D 챔피언 로직이 현재 활성 로직보다 유의미하게 우수할 때 (복합점수 10점 이상 상회)
        is_decayed = False
        decay_reason = ""

        active_id = active_logic.get("id")
        champion_id = champion.get("id")

        if last_week_return < 0.0:
            is_decayed = True
            decay_reason = f"주간 성적표 음수 손실 감쇠 발생 ({last_week_return:+.2f}%)"
        elif champion_id and champion_id != active_id:
            # 챔피언 로직이 다른 경우 성능 비교
            is_decayed = True
            decay_reason = f"Shadow R&D에서 더 높은 복합점수의 차세대 로직 발굴 (챔피언: {champion.get('name')})"

        evolution_event = None

        if is_decayed and champion and champion_id != active_id:
            # 자가 진화(Self-Evolution) 가동
            gen = state.get("generation", 1) + 1
            new_version = f"v1.{gen}.0"
            
            old_logic_name = active_logic.get("name", "Legacy Logic")
            new_logic = {
                "version": new_version,
                "id": champion["id"],
                "name": champion["name"],
                "dip_buy_pct": champion["parameters"]["dip_buy_pct"],
                "stop_loss_pct": champion["parameters"]["stop_loss_pct"],
                "hedge_switch_pct": champion["parameters"]["hedge_switch_pct"],
                "take_profit_pct": champion["parameters"]["take_profit_pct"],
                "description": champion["description"],
                "deployed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            evolution_event = {
                "triggered": True,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "generation": gen,
                "reason": decay_reason,
                "old_logic": f"{active_logic.get('version')} ({old_logic_name})",
                "new_logic": f"{new_version} ({new_logic['name']})",
                "new_params": new_logic,
                "champion_metrics": champion.get("metrics", {})
            }

            # 상태 갱신
            state["active_logic"] = new_logic
            state["generation"] = gen
            state["last_evolution_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_state(state)

            # 진화 로그 추가
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
            logs.append(evolution_event)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)

            return new_logic, evolution_event

        return active_logic, {
            "triggered": False,
            "status": "STABLE",
            "active_logic": active_logic,
            "reason": "현재 활성 로직이 최적 성능을 유지하고 있어 진화 보류."
        }
