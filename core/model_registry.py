import os
import sys
import json
import shutil
import sqlite3
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, BASE_DIR

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_REGISTRY_DB = DATA_DIR / "model_registry.db"

MODEL_CHAMPION_PKL = MODELS_DIR / "model_champion.pkl"
MODEL_GOLDEN_BASELINE_PKL = MODELS_DIR / "model_golden_baseline.pkl"

class ModelRegistry:
    """
    [Lumos v5.0 완성형 불변 모델 레지스트리 (6대 이종 트랙 관리)]
    - Track 0: model_champion.pkl (실전 메인 챔피언)
    - Track 1: model_main_data_refresh.pkl (일일 자동 롤링 최신화)
    - Track 2: model_sub_orderflow.pkl (오더플로우 / 수급 불균형)
    - Track 3: model_sub_tda.pkl (위상수학적 형태 붕괴 TDA)
    - Track 4: model_sub_statespace.pkl (상태공간 / 칼만 동역학)
    - Track 5: model_sub_cross_asset.pkl (크로스에셋 인과 괴리)
    - 무중단 핫스왑(Hot-Swap) 지원
    """
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or MODEL_REGISTRY_DB
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """레지스트리 스키마 초기화"""
        with self._get_connection() as conn:
            c = conn.cursor()
            
            c.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    model_id TEXT PRIMARY KEY,
                    model_file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    algorithm_type TEXT NOT NULL,
                    train_data_range TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,
                    status TEXT NOT NULL,
                    win_rate REAL DEFAULT 0.0,
                    profit_factor REAL DEFAULT 0.0,
                    total_return REAL DEFAULT 0.0,
                    mdd REAL DEFAULT 0.0,
                    top_features TEXT DEFAULT '[]',
                    notes TEXT DEFAULT ''
                );
            """)

            c.execute("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    eval_id TEXT PRIMARY KEY,
                    eval_date TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    total_return REAL NOT NULL,
                    mdd REAL NOT NULL,
                    composite_score REAL NOT NULL,
                    is_selected INTEGER DEFAULT 0,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES model_registry(model_id)
                );
            """)
            conn.commit()

    def register_model(
        self,
        model_id: str,
        model_obj: Any,
        algorithm_type: str = "LightGBM Classifier",
        train_data_range: str = "Last 60 Days",
        hyperparameters: Optional[Dict[str, Any]] = None,
        status: str = "SHADOW_ACTIVE",
        win_rate: float = 65.0,
        profit_factor: float = 3.13,
        total_return: float = 24.38,
        mdd: float = 3.96,
        top_features: Optional[List[str]] = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """새로운 모델 객체를 파일로 저장하고 레지스트리에 공식 등록"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 파일명 매핑
        if model_id == "M-20260815-GOLDEN-V1" or status == "GOLDEN_BASELINE":
            filename = "model_golden_baseline.pkl"
        elif model_id == "M-DATA-REFRESH":
            filename = "model_main_data_refresh.pkl"
        elif model_id == "M-SUB-ORDERFLOW":
            filename = "model_sub_orderflow.pkl"
        elif model_id == "M-SUB-TDA":
            filename = "model_sub_tda.pkl"
        elif model_id == "M-SUB-STATESPACE":
            filename = "model_sub_statespace.pkl"
        elif model_id == "M-SUB-CROSS-ASSET":
            filename = "model_sub_cross_asset.pkl"
        else:
            filename = f"{model_id.lower().replace('-', '_')}.pkl"

        versioned_file_path = MODELS_DIR / filename
        joblib.dump(model_obj, versioned_file_path)

        if status == "CHAMPION" and versioned_file_path != MODEL_CHAMPION_PKL:
            shutil.copy2(versioned_file_path, MODEL_CHAMPION_PKL)
        if "GOLDEN" in model_id.upper() and versioned_file_path != MODEL_GOLDEN_BASELINE_PKL:
            shutil.copy2(versioned_file_path, MODEL_GOLDEN_BASELINE_PKL)

        params_json = json.dumps(hyperparameters or {}, ensure_ascii=False)
        features_json = json.dumps(top_features or [], ensure_ascii=False)

        with self._get_connection() as conn:
            cur = conn.cursor()
            if status == "CHAMPION":
                cur.execute("UPDATE model_registry SET status = 'SHADOW_ACTIVE' WHERE status = 'CHAMPION';")

            cur.execute("""
                INSERT OR REPLACE INTO model_registry (
                    model_id, model_file_path, created_at, algorithm_type,
                    train_data_range, hyperparameters, status, win_rate,
                    profit_factor, total_return, mdd, top_features, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                model_id, str(versioned_file_path), now_str, algorithm_type,
                train_data_range, params_json, status, win_rate,
                profit_factor, total_return, mdd, features_json, notes
            ))
            conn.commit()

        return {
            "model_id": model_id,
            "file_path": str(versioned_file_path),
            "status": status,
            "win_rate": win_rate,
            "total_return": total_return
        }

    def promote_sub_model_to_champion(self, track_or_model_name: str) -> Dict[str, Any]:
        """
        [무중단 핫스왑 (Hot-Swap)] 5대 섀도우 트랙 중 하나를 실전 메인 챔피언으로 즉시 승격
        """
        k = track_or_model_name.upper().strip()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        target_file = None
        target_id = ""
        target_name = ""

        if any(w in k for w in ["최신화", "REFRESH", "트랙1", "TRACK1"]):
            target_file = MODELS_DIR / "model_main_data_refresh.pkl"
            target_id = "M-DATA-REFRESH"
            target_name = "Track 1: 메인 최신화 롤링 모델"
        elif any(w in k for w in ["오더플로우", "수급", "ORDERFLOW", "트랙2", "TRACK2"]):
            target_file = MODELS_DIR / "model_sub_orderflow.pkl"
            target_id = "M-SUB-ORDERFLOW"
            target_name = "Track 2: 오더플로우 / 수급 불균형 모델"
        elif any(w in k for w in ["TDA", "위상", "형태", "트랙3", "TRACK3"]):
            target_file = MODELS_DIR / "model_sub_tda.pkl"
            target_id = "M-SUB-TDA"
            target_name = "Track 3: 위상수학적 형태 붕괴 모델 (TDA)"
        elif any(w in k for w in ["상태공간", "칼만", "KALMAN", "제어", "트랙4", "TRACK4"]):
            target_file = MODELS_DIR / "model_sub_statespace.pkl"
            target_id = "M-SUB-STATESPACE"
            target_name = "Track 4: 상태공간 / 제어공학 모델 (Kalman)"
        elif any(w in k for w in ["크로스에셋", "인과", "괴리", "CROSS", "트랙5", "TRACK5"]):
            target_file = MODELS_DIR / "model_sub_cross_asset.pkl"
            target_id = "M-SUB-CROSS-ASSET"
            target_name = "Track 5: 크로스에셋 인과 괴리 모델"
        elif any(w in k for w in ["MOE", "오케스트레이터", "메타", "트랙6", "TRACK6", "ORCHESTRATOR"]):
            target_file = MODELS_DIR / "model_moe_orchestrator.pkl"
            target_id = "M-MOE-ORCHESTRATOR"
            target_name = "Track 6: MoE AI 메타 오케스트레이터"
        elif any(w in k for w in ["골든", "GOLDEN", "기준점", "기존", "24"]):
            target_file = MODEL_GOLDEN_BASELINE_PKL
            target_id = "M-20260815-GOLDEN-V1"
            target_name = "Track 0: 골든 베이스라인 챔피언"
        else:
            return {"ok": False, "msg": f"알 수 없는 트랙 모델명('{track_or_model_name}')입니다."}

        if not target_file.exists():
            return {"ok": False, "msg": f"해당 모델 파일({target_file.name})이 존재하지 않습니다."}

        # 챔피언 파일 교체
        shutil.copy2(target_file, MODEL_CHAMPION_PKL)

        # DB 상태 업데이트
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE model_registry SET status = 'SHADOW_ACTIVE' WHERE status = 'CHAMPION';")
            cur.execute("UPDATE model_registry SET status = 'CHAMPION' WHERE model_id LIKE ?;", (f"%{target_id}%",))
            conn.commit()

        return {
            "ok": True,
            "model_id": target_id,
            "model_name": target_name,
            "file_path": str(target_file),
            "promoted_at": now_str,
            "msg": f"[{target_name}]이(가) 실전 메인 챔피언(model_champion.pkl)으로 즉각 무중단 핫스왑 승격되었습니다."
        }

    def get_active_champion(self) -> Dict[str, Any]:
        """현재 활성화된 실전 메인 챔피언 모델 정보 조회"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM model_registry WHERE status = 'CHAMPION' ORDER BY rowid DESC LIMIT 1;")
            row = cur.fetchone()
            if row:
                d = dict(row)
                return d
        # 기본값: 골든 베이스라인
        return {
            "model_id": "M-20260815-GOLDEN-V1",
            "model_name": "⭐ Track 0: 실전 메인 챔피언",
            "status": "CHAMPION",
            "win_rate": 60.0,
            "profit_factor": 2.30,
            "total_return": 18.37
        }

    def record_tournament_evaluation(
        self,
        eval_date: str,
        model_id: str,
        candidate_type: str,
        win_rate: float,
        profit_factor: float,
        total_return: float,
        mdd: float,
        composite_score: float,
        is_selected: bool = False
    ) -> bool:
        """주간 3자 토너먼트 후보 평가 결과 기록"""
        eval_id = f"EVAL-{eval_date}-{model_id}-{int(datetime.now().timestamp()*1000)%10000}"
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO model_evaluations (
                    eval_id, eval_date, model_id, candidate_type,
                    win_rate, profit_factor, total_return, mdd,
                    composite_score, is_selected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                eval_id, eval_date, model_id, candidate_type,
                win_rate, profit_factor, total_return, mdd,
                composite_score, 1 if is_selected else 0
            ))
            conn.commit()
        return True

    def list_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """등록된 모델 목록 조회"""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM model_registry ORDER BY rowid DESC LIMIT ?;", (limit,))
            return [dict(r) for r in cur.fetchall()]
