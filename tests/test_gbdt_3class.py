import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml_engine import MLFeatureEngine
from core.moe_orchestrator import MoEMetaOrchestrator

class TestGBDT3ClassTripleBarrier(unittest.TestCase):
    """GBDT 3-Class Triple Barrier 및 MoE 연동 검증 단위 테스트"""

    def setUp(self):
        self.ml_engine = MLFeatureEngine(confidence_threshold=0.40)

    def test_1_triple_barrier_labeling_logic(self):
        """[검증 1] 경로 의존적 Triple Barrier (1, -1, 0) 라벨링 정확도 검증"""
        # 10개 캔들 생성
        # Bar 0: Close = 100
        # Bar 1~3: High hits 104 (+4%), Low hits 99 (-1%) -> 롱(+1) 조건 충족
        dates = pd.date_range("2026-08-24 09:30", periods=10, freq="15min")
        df_long = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [100.0, 101.0, 104.0, 102.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "Low":  [99.5,   99.5,  99.0,  99.0,  99.0,  99.0,  99.0,  99.0,  99.0,  99.0],
            "Close": [100.0, 101.0, 103.5, 101.5, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "Volume": [10000] * 10
        }, index=dates)

        labels_long = MLFeatureEngine.compute_triple_barrier_labels(df_long, take_profit=0.035, stop_loss=0.020, horizon=6)
        self.assertEqual(labels_long.iloc[0], 1, "Bar 0에서 롱(+3.5%) 도달 시 Label=1이어야 함")

        # Bar 0: Close = 100
        # Bar 1~3: Low hits 96 (-4%), High hits 101 (+1%) -> 숏(-1) 조건 충족
        df_short = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [100.0, 100.5, 100.8, 98.0, 97.0, 97.0, 97.0, 97.0, 97.0, 97.0],
            "Low":  [99.5,   98.0,  96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0, 96.0],
            "Close": [100.0,  98.5,  96.5, 97.0, 96.5, 96.5, 96.5, 96.5, 96.5, 96.5],
            "Volume": [10000] * 10
        }, index=dates)

        labels_short = MLFeatureEngine.compute_triple_barrier_labels(df_short, take_profit=0.035, stop_loss=0.020, horizon=6)
        self.assertEqual(labels_short.iloc[0], -1, "Bar 0에서 숏(-3.5%) 도달 시 Label=-1이어야 함")

        # 횡보 데이터 -> 0
        df_flat = pd.DataFrame({
            "Open": [100.0] * 10,
            "High": [100.5] * 10,
            "Low":  [99.5] * 10,
            "Close": [100.0] * 10,
            "Volume": [10000] * 10
        }, index=dates)
        labels_flat = MLFeatureEngine.compute_triple_barrier_labels(df_flat, take_profit=0.035, stop_loss=0.020, horizon=6)
        self.assertEqual(labels_flat.iloc[0], 0, "횡보 시 Label=0(관망)이어야 함")
        print("✅ [Test 1 통과] Triple Barrier 경로 의존성 3-Class (1, -1, 0) 라벨링 검증 완료")

    def test_2_model_training_and_proba(self):
        """[검증 2] 3-Class 다중 분류 학습 및 predict_proba 다차원 확률 출력 검증"""
        # 120개 캔들 생성
        dates = pd.date_range("2026-08-01 09:30", periods=150, freq="15min")
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.randn(150) * 0.5)
        df_dummy = pd.DataFrame({
            "Open": prices,
            "High": prices + np.random.rand(150) * 0.8,
            "Low": prices - np.random.rand(150) * 0.8,
            "Close": prices + np.random.randn(150) * 0.2,
            "Volume": np.random.randint(50000, 200000, 150)
        }, index=dates)

        model, top_10, top_3 = self.ml_engine.train_and_select_top_features(df_dummy)
        self.assertIsNotNone(model, "모델 학습이 완료되어야 함")
        self.assertEqual(len(top_10), 10, "Top 10 피처가 선별되어야 함")

        # add_confidence_columns 검증
        feat_df = self.ml_engine.extract_features(df_dummy)
        feat_df = self.ml_engine.add_confidence_columns(feat_df)
        self.assertIn("Prob_Long", feat_df.columns)
        self.assertIn("Prob_Short", feat_df.columns)
        self.assertIn("Prob_Neutral", feat_df.columns)
        self.assertIn("Direction", feat_df.columns)

        # predict_signal 검증
        sig, conf, reason = self.ml_engine.predict_signal(df_dummy)
        self.assertIn(sig, [1, -1, 0], "Signal 코드는 1, -1, 0 중 하나여야 함")
        self.assertTrue(0.0 <= conf <= 1.0, "확신도는 0과 1 사이여야 함")
        print("✅ [Test 2 통과] 3-Class 다중 분류 학습 및 확률 출력 검증 완료")

    def test_3_moe_integration(self):
        """[검증 3] MoE 오케스트레이터와 3-Class GBDT의 연동 무결성 검증"""
        moe = MoEMetaOrchestrator(confidence_threshold=0.75)
        dates = pd.date_range("2026-08-20 09:30", periods=50, freq="15min")
        df_dummy = pd.DataFrame({
            "Open": np.linspace(100, 105, 50),
            "High": np.linspace(101, 106, 50),
            "Low": np.linspace(99, 104, 50),
            "Close": np.linspace(100.5, 105.5, 50),
            "Volume": [100000] * 50
        }, index=dates)

        res = moe.evaluate_dual_filter_signal(df_dummy, threshold=0.75)
        self.assertIn("gating_confidence", res)
        self.assertIn("direction", res)
        self.assertIn("is_approved", res)
        self.assertIn("all_gating_confidences", res)
        print("✅ [Test 3 통과] MoE 오케스트레이터 및 3-Class GBDT 통합 연동 검증 완료")

if __name__ == "__main__":
    unittest.main()
