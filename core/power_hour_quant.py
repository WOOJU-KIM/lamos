import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import warnings
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

class PowerHourQuantEngine:
    """
    [Lumos Quant: 장 마감 전 파워 아워(14:30~15:30 EDT) 단기 퀀트 엔진]
    
    1. Volatility Expansion:
       - 5분봉 True Range vs ATR(14) 비율 (tr_ratio)
       - 단기 ATR(3봉) / 중기 ATR(9봉) 비율 (atr_ratio_short_long)
       - 5분봉 실현변동성(Realized Volatility) 3봉 vs 12봉 비율 (rv_ratio)
    
    2. Breakout:
       - 직전 6개봉(30m) 및 12개봉(60m) 고점/저점 돌파 (과거 [t-N : t-1] 엄격 분리, shift(1) 적용)
       - 당일 누적 High/Low 돌파 (당일 09:30~t-1 누적)
       - 거래량 서지(Volume Surge) & 변동성 확장 동반 확인
       
    3. VWAP:
       - 당일 09:30 리셋 누적 세션 VWAP과의 괴리율(%) 및 거리
       - 롤링 표준편차 기반 표준화 VWAP Z-score
       - VWAP 방향성 (+1 / -1)
       
    4. Strict Bias Prevention:
       - 피처 생성 시 미래봉 참조 0건 (Look-ahead bias 원천 차단)
       - 레이블은 미래 1~3봉(5m, 10m, 15m) 수익률을 별도 타깃(Y)으로만 사용
    """
    
    FEATURE_COLS = [
        'tr_ratio', 'atr_ratio_short_long', 'rv_3', 'rv_ratio',
        'vol_surge_6', 'vol_surge_12',
        'breakout_high_n6', 'breakout_low_n6',
        'breakout_day_high', 'breakout_day_low',
        'breakout_high_confirmed', 'breakout_low_confirmed',
        'vwap_dist_pct', 'vwap_zscore', 'vwap_dir',
        'body_to_range', 'ret_1bar', 'ret_3bar'
    ]

    def __init__(self, n_breakout_bars: int = 6):
        self.n_breakout_bars = n_breakout_bars
        self.model_meta_long = None
        self.model_meta_short = None
        self.model_direct = None

    def compute_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """
        5분봉 데이터프레임으로부터 파워 아워 전용 피처 및 미래 타깃 레이블 산출
        (Look-ahead bias 완전 배제: 과거 데이터는 shift(1) 또는 과거 롤링만 사용)
        """
        if df_5m is None or len(df_5m) < 30:
            return pd.DataFrame()

        df = df_5m.copy()
        
        # Datetime 및 세션 시간 파싱
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

        # 가격 기본 변수
        close = df['Close']
        high = df['High']
        low = df['Low']
        open_px = df['Open']
        vol = df['Volume']

        # [1. Volatility Expansion]
        # True Range
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
        df['tr'] = tr
        df['atr_14'] = tr.rolling(14).mean()
        df['tr_ratio'] = (tr / (df['atr_14'] + 1e-6)).fillna(1.0)
        
        # Short (3 bars = 15m) vs Long (9 bars = 45m) ATR ratio
        atr_3 = tr.rolling(3).mean()
        atr_9 = tr.rolling(9).mean()
        df['atr_ratio_short_long'] = (atr_3 / (atr_9 + 1e-6)).fillna(1.0)

        # Realized Volatility of 5m Log Returns
        log_ret = np.log(close / prev_close)
        df['rv_3'] = log_ret.rolling(3).std().fillna(0.0)
        df['rv_12'] = log_ret.rolling(12).std().fillna(1e-6)
        df['rv_ratio'] = (df['rv_3'] / (df['rv_12'] + 1e-6)).fillna(1.0)

        # [2. Volume Surge]
        vol_ma6 = vol.rolling(6).mean()
        vol_ma12 = vol.rolling(12).mean()
        df['vol_surge_6'] = (vol / (vol_ma6 + 1e-6)).fillna(1.0)
        df['vol_surge_12'] = (vol / (vol_ma12 + 1e-6)).fillna(1.0)

        # [3. Breakout - Strictly using past bars up to t-1]
        # Past N bars rolling max/min (excluding current bar)
        df['high_n6_prev'] = high.shift(1).rolling(self.n_breakout_bars).max()
        df['low_n6_prev'] = low.shift(1).rolling(self.n_breakout_bars).min()
        df['high_n12_prev'] = high.shift(1).rolling(12).max()
        df['low_n12_prev'] = low.shift(1).rolling(12).min()

        # Daily cumulative High/Low up to t-1
        df['day_high_prev'] = df.groupby('date_str')['High'].apply(lambda s: s.shift(1).cummax()).reset_index(level=0, drop=True)
        df['day_low_prev'] = df.groupby('date_str')['Low'].apply(lambda s: s.shift(1).cummin()).reset_index(level=0, drop=True)

        # Breakout signals
        df['breakout_high_n6'] = np.where(close >= df['high_n6_prev'], 1.0, 0.0)
        df['breakout_low_n6'] = np.where(close <= df['low_n6_prev'], 1.0, 0.0)
        df['breakout_day_high'] = np.where(close >= df['day_high_prev'], 1.0, 0.0)
        df['breakout_day_low'] = np.where(close <= df['day_low_prev'], 1.0, 0.0)

        # Confirmed Breakout: Breakout + Volume Surge (>= 1.2x) + Volatility Expansion (TR >= 1.1x ATR)
        df['breakout_high_confirmed'] = np.where(
            (df['breakout_high_n6'] == 1.0) & (df['vol_surge_6'] >= 1.2) & (df['tr_ratio'] >= 1.1), 1.0, 0.0
        )
        df['breakout_low_confirmed'] = np.where(
            (df['breakout_low_n6'] == 1.0) & (df['vol_surge_6'] >= 1.2) & (df['tr_ratio'] >= 1.1), 1.0, 0.0
        )

        # [4. VWAP - Cumulative Session VWAP with 09:30 Reset]
        df['cum_vol'] = df.groupby('date_str')['Volume'].cumsum()
        df['cum_pv'] = df.groupby('date_str').apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
        df['session_vwap'] = df['cum_pv'] / (df['cum_vol'] + 1e-6)

        df['vwap_dist_pct'] = ((close - df['session_vwap']) / (df['session_vwap'] + 1e-6)) * 100.0
        df['vwap_diff'] = close - df['session_vwap']
        df['vwap_diff_std'] = df['vwap_diff'].rolling(18).std().fillna(1e-6)
        df['vwap_zscore'] = (df['vwap_diff'] / (df['vwap_diff_std'] + 1e-6)).fillna(0.0)
        df['vwap_dir'] = np.where(close >= df['session_vwap'], 1.0, -1.0)

        # [5. Candle Structural Features]
        candle_range = (high - low).replace(0, 1e-6)
        df['body_to_range'] = ((close - open_px).abs() / candle_range).fillna(0.5)
        df['ret_1bar'] = (close / close.shift(1) - 1.0).fillna(0.0)
        df['ret_3bar'] = (close / close.shift(3) - 1.0).fillna(0.0)

        # [6. Target Labels - STRICTLY FORWARD LOOKING (NOT USED AS FEATURES)]
        # Forward returns for 5m (1 bar), 10m (2 bars), 15m (3 bars)
        df['fwd_ret_5m'] = (close.shift(-1) / close - 1.0)
        df['fwd_ret_10m'] = (close.shift(-2) / close - 1.0)
        df['fwd_ret_15m'] = (close.shift(-3) / close - 1.0)

        # Forward max high and min low over next 3 bars (15m window)
        next_high_3 = df[['High']].shift(-1).rolling(3).max().shift(-2)  # High of t+1..t+3
        # Alternative safe roll for forward high/low:
        fwd_high_3 = np.maximum(high.shift(-1), np.maximum(high.shift(-2), high.shift(-3)))
        fwd_low_3  = np.minimum(low.shift(-1), np.minimum(low.shift(-2), low.shift(-3)))
        df['fwd_max_up_15m'] = (fwd_high_3 / close - 1.0)
        df['fwd_max_down_15m'] = (fwd_low_3 / close - 1.0)

        # Meta-label targets: net profitable after 0.20% fee + slippage
        # Long profitable if 15m return > +0.50%
        df['target_meta_long'] = np.where(df['fwd_ret_15m'] >= 0.005, 1, 0)
        # Short profitable if 15m return < -0.50%
        df['target_meta_short'] = np.where(df['fwd_ret_15m'] <= -0.005, 1, 0)

        # Direct Direction Classification Target: 1 (Long), 2 (Short), 0 (Flat)
        cond_long = df['fwd_ret_15m'] >= 0.008
        cond_short = df['fwd_ret_15m'] <= -0.008
        df['target_direct_dir'] = np.where(cond_long, 1, np.where(cond_short, 2, 0))

        return df

    def evaluate_rule_signal(self, row: pd.Series) -> str:
        """
        룰 기반 진입 신호 생성 (14:30 ~ 15:30 EDT 전용)
        반환값: 'LONG', 'SHORT', 'NONE'
        """
        t_str = row['time_str']
        if not ('14:30' <= t_str <= '15:30'):
            return 'NONE'

        # Long 조건:
        # 1. Volatility Expansion: tr_ratio >= 1.15 또는 atr_ratio_short_long >= 1.10
        # 2. Volume Surge: vol_surge_6 >= 1.25
        # 3. Breakout: breakout_high_n6 == 1 또는 breakout_day_high == 1
        # 4. VWAP: Close >= session_vwap (vwap_dir == 1.0 및 vwap_zscore > 0.0)
        is_vol_long = (row['tr_ratio'] >= 1.15) or (row['atr_ratio_short_long'] >= 1.10)
        is_vol_surge = (row['vol_surge_6'] >= 1.25)
        is_breakout_long = (row['breakout_high_n6'] == 1.0) or (row['breakout_day_high'] == 1.0)
        is_vwap_long = (row['vwap_dir'] == 1.0) and (row['vwap_zscore'] > 0.0)

        if is_vol_long and is_vol_surge and is_breakout_long and is_vwap_long:
            return 'LONG'

        # Short 조건:
        # 1. Volatility Expansion: tr_ratio >= 1.15 또는 atr_ratio_short_long >= 1.10
        # 2. Volume Surge: vol_surge_6 >= 1.25
        # 3. Breakout: breakout_low_n6 == 1 또는 breakout_day_low == 1
        # 4. VWAP: Close <= session_vwap (vwap_dir == -1.0 및 vwap_zscore < 0.0)
        is_breakout_short = (row['breakout_low_n6'] == 1.0) or (row['breakout_day_low'] == 1.0)
        is_vwap_short = (row['vwap_dir'] == -1.0) and (row['vwap_zscore'] < 0.0)

        if is_vol_long and is_vol_surge and is_breakout_short and is_vwap_short:
            return 'SHORT'

        return 'NONE'

    def train_models(self, train_df: pd.DataFrame, min_samples: int = 50):
        """
        LightGBM 모델 훈련 (Train 세트 전용)
        1) Meta-labeling GBDT: Rule 신호 발생 시 기대값(승률) 필터링용
        2) Direct GBDT: 룰 신호 없이 피처만으로 직접 방향 예측용
        """
        # 파워 아워 구간(14:30~15:30) 데이터만 필터링
        ph_train = train_df[(train_df['time_str'] >= '14:30') & (train_df['time_str'] <= '15:30')].dropna(subset=self.FEATURE_COLS + ['target_meta_long', 'target_meta_short'])
        
        if len(ph_train) < min_samples:
            # Fallback: 전체 세션 오후 데이터 포함하여 안정적 훈련
            ph_train = train_df[(train_df['time_str'] >= '13:30') & (train_df['time_str'] <= '15:30')].dropna(subset=self.FEATURE_COLS + ['target_meta_long', 'target_meta_short'])

        X = ph_train[self.FEATURE_COLS]

        # 1. Meta-Labeling Model for Long
        y_long = ph_train['target_meta_long']
        self.model_meta_long = LGBMClassifier(
            n_estimators=60,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model_meta_long.fit(X, y_long)

        # 2. Meta-Labeling Model for Short
        y_short = ph_train['target_meta_short']
        self.model_meta_short = LGBMClassifier(
            n_estimators=60,
            max_depth=3,
            num_leaves=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model_meta_short.fit(X, y_short)

        # 3. Direct Direction Model (Multi-class: 0=Flat, 1=Long, 2=Short)
        y_direct = ph_train['target_direct_dir']
        self.model_direct = LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            num_leaves=11,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model_direct.fit(X, y_direct)

    def predict_meta_long(self, row: pd.Series) -> float:
        """Rule Long 신호에 대한 승률 확률 (0.0 ~ 1.0)"""
        if self.model_meta_long is None:
            return 0.5
        x = pd.DataFrame([row[self.FEATURE_COLS]])
        prob = self.model_meta_long.predict_proba(x)[0, 1]
        return float(prob)

    def predict_meta_short(self, row: pd.Series) -> float:
        """Rule Short 신호에 대한 승률 확률 (0.0 ~ 1.0)"""
        if self.model_meta_short is None:
            return 0.5
        x = pd.DataFrame([row[self.FEATURE_COLS]])
        prob = self.model_meta_short.predict_proba(x)[0, 1]
        return float(prob)

    def predict_direct(self, row: pd.Series, threshold: float = 0.50) -> str:
        """GBDT Only: 직접 방향 예측"""
        if self.model_direct is None:
            return 'NONE'
        t_str = row['time_str']
        if not ('14:30' <= t_str <= '15:30'):
            return 'NONE'
            
        x = pd.DataFrame([row[self.FEATURE_COLS]])
        probs = self.model_direct.predict_proba(x)[0]
        # classes: [0 (Flat), 1 (Long), 2 (Short)]
        classes = self.model_direct.classes_
        prob_dict = {c: p for c, p in zip(classes, probs)}
        p_long = prob_dict.get(1, 0.0)
        p_short = prob_dict.get(2, 0.0)

        if p_long >= threshold and p_long > p_short:
            return 'LONG'
        elif p_short >= threshold and p_short > p_long:
            return 'SHORT'
        return 'NONE'
