import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator
from core.data_lake import MarketDataLake

lake = MarketDataLake()
soxl_15m = lake.load_candles("SOXL", "15m")
moe = MoEMetaOrchestrator(confidence_threshold=0.55, gbdt_threshold=0.55, mode="hybrid_v3")

idx = 600
full_f = moe.gbdt_engine.extract_features(soxl_15m)
full_f = moe.gbdt_engine.add_confidence_columns(full_f)
print(f"Full series: dir={full_f['Direction'].iloc[idx]}, conf={full_f['Confidence'].iloc[idx]:.4f}")

for window_len in [60, 100, 200, 300, 500, 601]:
    sub = soxl_15m.iloc[max(0, idx-window_len+1):idx+1].copy()
    f = moe.gbdt_engine.extract_features(sub)
    f = moe.gbdt_engine.add_confidence_columns(f)
    print(f"Window {window_len:3d}: dir={f['Direction'].iloc[-1]}, conf={f['Confidence'].iloc[-1]:.4f}")
