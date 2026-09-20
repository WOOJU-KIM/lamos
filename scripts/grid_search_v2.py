import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
import itertools
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel
import config

def run_grid():
    lake = MarketDataLake()
    print('Loading data...')
    long_15m = lake.load_candles(config.TICKER_LONG, '15m')
    short_15m = lake.load_candles(config.TICKER_SHORT, '15m')
    trend_60m = lake.load_candles(config.TICKER_TREND, '60m')
    
    long_5m = lake.load_candles(config.TICKER_LONG, '5m')
    short_5m = lake.load_candles(config.TICKER_SHORT, '5m')
    
    if len(long_15m) < 1000:
        return
        
    print('Preparing ML Features...')
    engine = MLFeatureEngine()
    long_feat = engine.extract_features(long_15m)
    short_feat = engine.extract_features(short_15m)
    
    # Pre-train just once to get predictions
    from lightgbm import LGBMClassifier
    train_end = len(long_feat) - 2000
    train_df = long_feat.iloc[:train_end].copy()
    train_df = train_df.dropna()
    
    labels = MLFeatureEngine.compute_triple_barrier_labels(train_df, take_profit=0.012, stop_loss=0.008, horizon=4)
    train_df['Label'] = labels
    
    feature_cols = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c]) and c not in ['Label']]
    
    clf = LGBMClassifier(n_estimators=120, learning_rate=0.02, colsample_bytree=0.5, random_state=42)
    clf.fit(train_df[feature_cols], train_df['Label'])
    
    # Predict all out of sample
    test_df = long_feat.iloc[train_end:].copy().dropna()
    probs = clf.predict_proba(test_df[feature_cols])
    
    pl = probs[:, 2]
    ps = probs[:, 0]
    pn = probs[:, 1]
    
    w_confs = np.full(len(test_df), 0.50, dtype=float)
    w_dirs = ['NONE'] * len(test_df)
    
    for i in range(len(test_df)):
        if pl[i] > pn[i] and pl[i] > ps[i]:
            w_confs[i] = min(0.95, max(0.50, 0.50 + (pl[i]-0.333)*1.15))
            w_dirs[i] = f'LONG_{config.TICKER_LONG}'
        elif ps[i] > pn[i] and ps[i] > pl[i]:
            w_confs[i] = min(0.95, max(0.50, 0.50 + (ps[i]-0.333)*1.15))
            w_dirs[i] = f'SHORT_{config.TICKER_SHORT}'
            
    test_df['Confidence'] = w_confs
    test_df['Direction'] = w_dirs
    
    # Set up grid
    param_grid = {
        'conf_hurdle': [0.55, 0.58, 0.60],
        'trail_trigger': [0.008, 0.010],
        'trail_pullback': [0.002, 0.003],
        'time_stop_bars': [12, 18], # 60m, 90m in 5m bars
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    # Pre-filter dataframes
    day_long_5 = long_5m[long_5m['datetime'] >= test_df['datetime'].iloc[0]].reset_index(drop=True)
    day_short_5 = short_5m[short_5m['datetime'] >= test_df['datetime'].iloc[0]].reset_index(drop=True)
    
    print(f'Starting Grid Search with {len(combinations)} combinations...')
    
    for idx, params in enumerate(combinations):
        current_capital = 10_000_000
        trades_cnt = 0
        wins_cnt = 0
        long_cnt = 0
        short_cnt = 0
        
        for i, row in test_df.iterrows():
            curr_dt = row['datetime']
            dir_gbdt = row['Direction']
            conf_gbdt = row['Confidence']
            
            if dir_gbdt == 'NONE' or conf_gbdt < params['conf_hurdle']:
                continue
                
            is_long = (dir_gbdt == f'LONG_{config.TICKER_LONG}')
            
            # Simplified entry
            entry_price = float(row['Close'])
            
            target_5m = day_long_5 if is_long else day_short_5
            post_5m = target_5m[target_5m['datetime'] > curr_dt]
            
            if post_5m.empty: continue
            
            # Trailing stop logic
            hard_sl_px = entry_price * 0.99  # -1.0%
            max_tp_px = entry_price * 1.05   # Max +5.0%
            trigger_px = entry_price * (1.0 + params['trail_trigger'])
            pullback_pct = params['trail_pullback']
            
            highest_px = entry_price
            trailing_active = False
            
            exit_price = None
            
            max_eval = min(len(post_5m), params['time_stop_bars'])
            for k in range(max_eval):
                bar = post_5m.iloc[k]
                b_h = float(bar['High'])
                b_l = float(bar['Low'])
                b_c = float(bar['Close'])
                
                # Check Hard SL first
                if b_l <= hard_sl_px:
                    exit_price = hard_sl_px
                    break
                    
                # Check Max TP
                if b_h >= max_tp_px:
                    exit_price = max_tp_px
                    break
                    
                # Update highest
                if b_h > highest_px:
                    highest_px = b_h
                    
                # Activate trailing?
                if highest_px >= trigger_px:
                    trailing_active = True
                    
                if trailing_active:
                    trail_sl_px = highest_px * (1.0 - pullback_pct)
                    if b_l <= trail_sl_px:
                        exit_price = trail_sl_px
                        break
                        
                # Time Stop
                if k == max_eval - 1:
                    exit_price = b_c
                    break
            
            if exit_price is None:
                exit_price = entry_price
                
            raw_ret = (exit_price / entry_price) - 1.0
            net_ret = raw_ret - 0.0020  # 0.2% round trip
            
            current_capital *= (1.0 + net_ret)
            trades_cnt += 1
            if net_ret > 0: wins_cnt += 1
            if is_long: long_cnt += 1
            else: short_cnt += 1
            
        results.append({
            'Conf': params['conf_hurdle'],
            'Trig': params['trail_trigger'],
            'Pull': params['trail_pullback'],
            'Time': params['time_stop_bars']*5,
            'FinalCap': current_capital,
            'Trades': trades_cnt,
            'WinRate': (wins_cnt/trades_cnt*100) if trades_cnt else 0,
            'Longs': long_cnt,
            'Shorts': short_cnt
        })
        
    res_df = pd.DataFrame(results).sort_values(by='FinalCap', ascending=False)
    print(res_df.head(10).to_markdown(index=False))

if __name__ == '__main__':
    run_grid()
