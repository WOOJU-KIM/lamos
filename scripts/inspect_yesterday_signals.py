import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine

lake = MarketDataLake()
tqqq_15m = lake.load_candles("TQQQ", "15m")
nvda_15m = lake.load_candles("NVDA", "15m")
qqq_15m = lake.load_candles("QQQ", "15m")
vix_15m = lake.load_candles("^VIX", "15m")
soxx_60m = lake.load_candles("SOXX", "60m")

print("=== TQQQ 최근 5개 15분봉 ===")
print(tqqq_15m[["Open", "High", "Low", "Close", "Volume"]].tail(5))

c = tqqq_15m["Close"]
ret_5 = float((c.iloc[-1] / c.iloc[-5] - 1.0))
print(f"\nTQQQ 5봉 수익률: {ret_5*100:.2f}%")

# 1. 크로스에셋 결과
cross_model = CrossAssetDislocationModel()
n_c = nvda_15m["Close"]
q_c = qqq_15m["Close"]
v_c = vix_15m["Close"]
soxx_15m = lake.load_candles("SOXX", "15m")
sx_c = soxx_15m["Close"] if not soxx_15m.empty else n_c
soxx_r = float(sx_c.iloc[-1] / sx_c.iloc[-5] - 1.0) if len(sx_c) >= 5 else nvda_r

sig_code, exp_conf, meta = cross_model.predict_signal(
    tqqq_ret=ret_5,
    nvda_ret=nvda_r,
    soxx_ret=soxx_r,
    qqq_ret=qqq_r,
    vix_ret=vix_r,
    tnx_ret=0.0
)
print(f"\n[크로스에셋 괴리 모델] sig_code={sig_code}, exp_conf={exp_conf}")
print("Meta:", meta)

# 2. GBDT 결과
gbdt = MLFeatureEngine()
df_feat = gbdt.extract_features(tqqq_15m)
last_feat = df_feat.iloc[-1]
print(f"\n[GBDT 모델 피처]")
print(f"Confidence: {last_feat.get('Confidence')}")
print(f"VWAP_Diff: {last_feat.get('VWAP_Diff')}")
print(f"RSI_14: {last_feat.get('RSI_14')}")
print(f"BB_Lower: {last_feat.get('BB_Lower')}")
print(f"Close: {last_feat.get('Close')}")

# 3. 60분봉 추세
if not soxx_60m.empty:
    soxx_ema20 = soxx_60m['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
    soxx_c = soxx_60m['Close'].iloc[-1]
    print(f"\n[60분봉 추세] SOXX Close: {soxx_c:.2f}, EMA20: {soxx_ema20:.2f} (차이: {((soxx_c/soxx_ema20)-1)*100:.2f}%)")

moe = MoEMetaOrchestrator(confidence_threshold=0.75)
res = moe.evaluate_dual_filter_signal(tqqq_15m, threshold=0.75)
print("\n=== 최종 MoE 평가 결과 ===")
print(json.dumps(res, indent=2, ensure_ascii=False))
