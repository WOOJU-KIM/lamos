import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lamos")
sys.path.insert(0, str(PROJECT_ROOT))

import config
import joblib
import pandas as pd
from datetime import datetime, timedelta

from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator, MOE_MODEL_PATH

def retrain_weekly_v4():
    print("🚀 [V4 최적화 모델 주간 롤링 학습 시작]")
    lake = MarketDataLake()
    
    long_15m = lake.load_candles(config.TICKER_LONG, "15m")
    if long_15m.empty:
        print("Error: No data found.")
        return
    end_date_str = long_15m['datetime'].max()
    
    end_dt = pd.to_datetime(end_date_str)
    start_dt = end_dt - timedelta(days=730)
    start_date_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    
    long_15m = long_15m[long_15m['datetime'] >= start_date_str].copy()
    long_15m = long_15m.sort_values('datetime').reset_index(drop=True)
    
    print(f"✅ 학습 데이터 기간: {long_15m['datetime'].min()} ~ {long_15m['datetime'].max()}")
    print(f"✅ 학습 데이터 캔들 수: {len(long_15m)}개")
    
    orchestrator = MoEMetaOrchestrator(confidence_threshold=0.62, gbdt_threshold=0.62, mode="hybrid_v3")
    print("⏳ GBDT 모델 학습 중...")
    orchestrator.gbdt_engine.train_and_select_top_features(long_15m)
    
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    champ_path = model_dir / "model_champion.pkl"
    
    joblib.dump(orchestrator, MOE_MODEL_PATH)
    joblib.dump(orchestrator, champ_path)
    print(f"🎯 [학습 완료] V4 챔피언 모델이 저장되었습니다: {champ_path}")

if __name__ == "__main__":
    retrain_weekly_v4()
