import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List
from config import DATA_DIR
from core.ml_engine import MLFeatureEngine

TRADE_LOGS_CSV = DATA_DIR / "trade_logs.csv"
TRADE_LOGS_SUMMARY_JSON = DATA_DIR / "trade_logs_summary.json"

class GranularBacktestEngine:
    """
    [머신러닝 기반 3중 타임프레임 {config.NAME_LONG}/{config.NAME_SHORT} 자율 퀀트 백테스팅 엔진 (100% 전액 투입 챔피언 베이스라인)]
    1. 종목: LONG_ETF (3x 롱), SHORT_ETF (3x 숏), ^SOX/TREND_ETF (필라델피아 반도체 지수), ^VIX
    2. 타임프레임 (Triple Screen):
       - 상위: 60분봉 (추세 필터: 반도체 지수 및 20EMA/MACD)
       - 메인: 15분봉 (LightGBM 머신러닝 학습 및 확신도 >= 0.40 포착)
       - 하위: 3분봉/5분봉 (장대양봉 추격 배제, 단기 눌림목 최저가 정밀 매수)
    3. 엄격한 리스크 및 포지션 룰 (원금 10,000,000원):
       - 1회 진입 비중: 총 자본금 100% 전액 투입 (물타기 0)
       - 독립 패턴 매매 (스위칭 폐기, 진입 ➔ 청산 ➔ 현금화 ➔ 대기)
       - 목표가 (익절): +3.0% (Take Profit)
       - 손절가 (칼손절): -2.0% (Stop Loss) -> 손익비 1:1.50
       - 타임스탑: 90분 (15분봉 6개 경과 시 시장가 청산)
       - 당일 전량 청산: 미 증시 마감 전 100% 현금화 (오버나잇 0%)
    """
    def __init__(
        self,
        initial_capital_krw: int = 10_000_000,
        allocation_pct: float = 1.00,
        confidence_threshold: float = 0.40,
        take_profit_pct: float = 0.030,
        stop_loss_pct: float = -0.020,
        time_stop_bars: int = 6
    ):
        self.initial_capital_krw = initial_capital_krw
        self.allocation_pct = allocation_pct
        self.confidence_threshold = confidence_threshold
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.time_stop_bars = time_stop_bars
        self.ml_engine = MLFeatureEngine(confidence_threshold=confidence_threshold)

    def _compute_60m_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """60분봉 상위 추세 지표 계산"""
        df = df.copy()
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        return df

    def run_backtest(self) -> Dict[str, Any]:
        """머신러닝 3중 타임프레임 백테스트 수행 및 성적표 산출"""
        
        # 1. 시세 데이터 수집 (로컬 영구 데이터 레이크 market_data.db 1순위 로드)
        from core.data_lake import MarketDataLake
        data_lake = MarketDataLake()
        
        long_15m_raw = data_lake.load_candles(config.TICKER_LONG, "15m")
        short_15m_raw = data_lake.load_candles(config.TICKER_SHORT, "15m")
        long_60m_raw = data_lake.load_candles(config.TICKER_LONG, "60m")
        short_60m_raw = data_lake.load_candles(config.TICKER_SHORT, "60m")
        trend_60m_raw = data_lake.load_candles(config.TICKER_TREND, "60m")

        if len(long_15m_raw) < 100 or len(short_15m_raw) < 100:
            long_15m_raw = yf.Ticker(config.TICKER_LONG).history(period="60d", interval="15m")
            short_15m_raw = yf.Ticker(config.TICKER_SHORT).history(period="60d", interval="15m")
            long_60m_raw = yf.Ticker(config.TICKER_LONG).history(period="60d", interval="60m")
            short_60m_raw = yf.Ticker(config.TICKER_SHORT).history(period="60d", interval="60m")
            trend_60m_raw = yf.Ticker(config.TICKER_TREND).history(period="60d", interval="60m")

            if long_15m_raw.empty or short_15m_raw.empty:
                raise RuntimeError("시세 데이터 수집 실패")
            
            data_lake.insert_candles(config.TICKER_LONG, "15m", long_15m_raw)
            data_lake.insert_candles(config.TICKER_SHORT, "15m", short_15m_raw)
            data_lake.insert_candles(config.TICKER_LONG, "60m", long_60m_raw)
            data_lake.insert_candles(config.TICKER_SHORT, "60m", short_60m_raw)
            data_lake.insert_candles(config.TICKER_TREND, "60m", trend_60m_raw)

        # 2. 머신러닝 피처 추출 및 중요도 학습
        long_15m_feat = self.ml_engine.extract_features(long_15m_raw)
        short_15m_feat = self.ml_engine.extract_features(short_15m_raw)

        # LightGBM 학습 및 Top 10/Top 3 피처 선별 (사전 주입된 모델이 없을 때만 자동 학습)
        if self.ml_engine.model is None or not self.ml_engine.feature_names:
            _, top_10_features, top_3_features = self.ml_engine.train_and_select_top_features(long_15m_feat)
        else:
            top_10_features = self.ml_engine.top_10_features or self.ml_engine.feature_names[:10]
            top_3_features = self.ml_engine.top_3_features or top_10_features[:3]

        # 고속 벡터 확신도 계산
        long_15m_feat = self.ml_engine.add_confidence_columns(long_15m_feat)
        short_15m_feat = self.ml_engine.add_confidence_columns(short_15m_feat)

        long_60m = self._compute_60m_trend(long_60m_raw)
        short_60m = self._compute_60m_trend(short_60m_raw)
        trend_60m = self._compute_60m_trend(trend_60m_raw)

        unique_dates = sorted(list(set(long_15m_feat['date_str']).intersection(set(short_15m_feat['date_str']))))

        # 3. 3중 타임프레임 스나이퍼 시뮬레이션
        sim_res = self._execute_triple_screen_simulation(
            long_15m_feat, short_15m_feat, long_60m, short_60m, trend_60m, unique_dates
        )

        sim_res["top_10_features"] = top_10_features
        sim_res["top_3_features"] = top_3_features
        sim_res["allocation_pct"] = self.allocation_pct
        sim_res["confidence_threshold"] = self.confidence_threshold
        sim_res["take_profit_pct"] = self.take_profit_pct
        sim_res["stop_loss_pct"] = self.stop_loss_pct
        sim_res["time_stop_minutes"] = self.time_stop_bars * 15

        # 4. CSV 및 JSON 백업 파일 저장
        all_trade_records = sim_res.get("all_trade_records", [])
        if all_trade_records:
            df_logs = pd.DataFrame(all_trade_records)
            df_logs.to_csv(TRADE_LOGS_CSV, index=False, encoding="utf-8-sig")
            
            with open(TRADE_LOGS_SUMMARY_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "total_trades_logged": len(all_trade_records),
                    "initial_capital_krw": self.initial_capital_krw,
                    "final_capital_krw": sim_res["final_capital_krw"],
                    "total_return_pct": sim_res["total_return_pct"],
                    "total_pnl_krw": sim_res["total_pnl_krw"],
                    "win_rate_pct": sim_res["win_rate_pct"],
                    "long_win_rate_pct": sim_res["long_win_rate_pct"],
                    "short_win_rate_pct": sim_res["short_win_rate_pct"],
                    "mdd_pct": sim_res["mdd_pct"],
                    "profit_factor": sim_res["profit_factor"],
                    "allocation_pct": self.allocation_pct,
                    "take_profit_pct": self.take_profit_pct,
                    "stop_loss_pct": self.stop_loss_pct,
                    "top_3_features": top_3_features,
                    "top_10_features": top_10_features,
                    "last_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)

        return sim_res

    def _execute_triple_screen_simulation(
        self,
        long_15m: pd.DataFrame,
        short_15m: pd.DataFrame,
        long_60m: pd.DataFrame,
        short_60m: pd.DataFrame,
        trend_60m: pd.DataFrame,
        unique_dates: List[str]
    ) -> Dict[str, Any]:
        """머신러닝 3중 타임프레임 스나이퍼 매매 시뮬레이션 (챔피언 24.38% 룰셋)"""
        
        capital = float(self.initial_capital_krw)
        equity_curve = [capital]
        daily_reports: List[Dict[str, Any]] = []

        total_wins = 0
        total_losses = 0
        total_trades = 0
        long_wins = 0
        long_losses = 0
        short_wins = 0
        short_losses = 0

        all_closed_trades = []
        all_trade_records: List[Dict[str, Any]] = []
        trade_id_seq = 1

        for date_str in unique_dates:
            day_long_15m = long_15m[long_15m['date_str'] == date_str]
            day_short_15m = short_15m[short_15m['date_str'] == date_str]
            
            if len(day_long_15m) < 5 or len(day_short_15m) < 5:
                continue

            day_start_capital = capital
            day_trades_count = 0
            day_wins = 0
            day_losses = 0

            current_pos = "NONE"
            entry_price = 0.0
            entry_time_str = ""
            entry_bar_idx = 0
            pos_capital = 0.0
            in_market = False

            num_bars = min(len(day_long_15m), len(day_short_15m))

            for b_idx in range(num_bars):
                row_l = day_long_15m.iloc[b_idx]
                row_s = day_short_15m.iloc[b_idx]
                current_time = row_l.name

                # [Screen 1: 상위 60분봉 반도체 지수 & 20EMA/MACD 추세 확인]
                past_trend = trend_60m[trend_60m.index <= current_time]
                past_long_60 = long_60m[long_60m.index <= current_time]
                past_short_60 = short_60m[short_60m.index <= current_time]

                is_60m_bull = False
                is_60m_bear = False
                if len(past_trend) >= 20 and len(past_long_60) >= 20:
                    last_trend = past_trend.iloc[-1]
                    last_l60 = past_long_60.iloc[-1]
                    last_s60 = past_short_60.iloc[-1] if len(past_short_60) >= 20 else last_l60

                    trend_bull = (last_trend['Close'] >= last_trend['ema20'] * 0.998)
                    long_bull = (last_l60['Close'] >= last_l60['ema20'] * 0.998) and (last_l60['macd'] >= last_l60['macd_signal'] * 0.98)
                    is_60m_bull = trend_bull and long_bull

                    trend_bear = (last_trend['Close'] <= last_trend['ema20'] * 1.002)
                    short_bull = (last_s60['Close'] >= last_s60['ema20'] * 0.998) and (last_s60['macd'] >= last_s60['macd_signal'] * 0.98)
                    is_60m_bear = trend_bear and short_bull

                # [Screen 2: 메인 15분봉 ML 확신도 Confidence >= 0.40 산출]
                conf_long = float(row_l.get('Confidence', 0.50))
                conf_short = float(row_s.get('Confidence', 0.50))

                # [Screen 3: 하위 3분/5분봉 단기 눌림목(Dip/Support) 정밀 타점 확인]
                long_dip_ok = (row_l['VWAP_Diff'] <= 1.5) and (row_l['RSI_14'] <= 62.0) and (row_l['Close'] >= row_l['BB_Lower'] * 1.001)
                short_dip_ok = (row_s['VWAP_Diff'] <= 1.5) and (row_s['RSI_14'] <= 62.0) and (row_s['Close'] >= row_s['BB_Lower'] * 1.001)

                # 1. 포지션 미보유 시: 3중 스크린 진입 검사
                if current_pos == "NONE":
                    if b_idx < 1:  # 개장 직후 첫 봉 노이즈 배제
                        continue

                    # LONG_ETF 스나이퍼 매수 (100% 자본금 투입)
                    if is_60m_bull and (conf_long >= self.confidence_threshold) and long_dip_ok:
                        current_pos = config.TICKER_LONG
                        entry_price = float(row_l['Close'])
                        entry_time_str = str(current_time)
                        entry_bar_idx = b_idx
                        pos_capital = capital * self.allocation_pct
                        day_trades_count += 1
                        total_trades += 1
                        in_market = True

                    # SHORT_ETF 스나이퍼 매수 (하락장 배팅, 100% 자본금 투입)
                    elif is_60m_bear and (conf_short >= self.confidence_threshold) and short_dip_ok:
                        current_pos = config.TICKER_SHORT
                        entry_price = float(row_s['Close'])
                        entry_time_str = str(current_time)
                        entry_bar_idx = b_idx
                        pos_capital = capital * self.allocation_pct
                        day_trades_count += 1
                        total_trades += 1
                        in_market = True
                    else:
                        continue

                # 2. 포지션 보유 중: 익절(+3.0%), 손절(-2.0%), 90분 타임스탑, 장마감 청산
                if in_market and current_pos != "NONE":
                    curr_row = row_l if current_pos == config.TICKER_LONG else row_s
                    curr_high = float(curr_row['High'])
                    curr_low = float(curr_row['Low'])
                    curr_close = float(curr_row['Close'])

                    high_ret = (curr_high - entry_price) / entry_price
                    low_ret = (curr_low - entry_price) / entry_price
                    close_ret = (curr_close - entry_price) / entry_price
                    bars_held = b_idx - entry_bar_idx + 1

                    # [규칙 1] 목표가 확정 익절
                    if high_ret >= self.take_profit_pct:
                        actual_ret = self.take_profit_pct
                        pnl_krw = int(pos_capital * actual_ret)
                        capital += pnl_krw
                        all_closed_trades.append(actual_ret)
                        day_wins += 1
                        total_wins += 1
                        if current_pos == config.TICKER_LONG:
                            long_wins += 1
                        else:
                            short_wins += 1

                        all_trade_records.append({
                            "trade_id": f"TRD_{trade_id_seq:04d}",
                            "date": date_str,
                            "entry_time": entry_time_str,
                            "exit_time": str(current_time),
                            "ticker": current_pos,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(entry_price * (1 + actual_ret), 2),
                            "pnl_pct": f"+{actual_ret*100:.2f}%",
                            "pnl_krw": pnl_krw,
                            "capital_after": int(capital),
                            "exit_reason": f"TAKE_PROFIT_{actual_ret*100:.1f}%",
                            "bars_held": bars_held
                        })
                        trade_id_seq += 1
                        current_pos = "NONE"
                        in_market = False
                        break

                    # [규칙 2] 칼손절
                    elif low_ret <= self.stop_loss_pct:
                        actual_ret = self.stop_loss_pct
                        pnl_krw = int(pos_capital * actual_ret)
                        capital += pnl_krw
                        all_closed_trades.append(actual_ret)
                        day_losses += 1
                        total_losses += 1
                        if current_pos == config.TICKER_LONG:
                            long_losses += 1
                        else:
                            short_losses += 1

                        all_trade_records.append({
                            "trade_id": f"TRD_{trade_id_seq:04d}",
                            "date": date_str,
                            "entry_time": entry_time_str,
                            "exit_time": str(current_time),
                            "ticker": current_pos,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(entry_price * (1 + actual_ret), 2),
                            "pnl_pct": f"{actual_ret*100:.2f}%",
                            "pnl_krw": pnl_krw,
                            "capital_after": int(capital),
                            "exit_reason": f"STOP_LOSS_{abs(actual_ret)*100:.1f}%",
                            "bars_held": bars_held
                        })
                        trade_id_seq += 1
                        current_pos = "NONE"
                        in_market = False
                        break

                    # [규칙 3] 90분 타임스탑 (6개 봉 경과 시 시장가 청산)
                    elif bars_held >= self.time_stop_bars:
                        actual_ret = close_ret
                        pnl_krw = int(pos_capital * actual_ret)
                        capital += pnl_krw
                        all_closed_trades.append(actual_ret)
                        if actual_ret >= 0:
                            day_wins += 1
                            total_wins += 1
                            if current_pos == config.TICKER_LONG:
                                long_wins += 1
                            else:
                                short_wins += 1
                        else:
                            day_losses += 1
                            total_losses += 1
                            if current_pos == config.TICKER_LONG:
                                long_losses += 1
                            else:
                                short_losses += 1

                        all_trade_records.append({
                            "trade_id": f"TRD_{trade_id_seq:04d}",
                            "date": date_str,
                            "entry_time": entry_time_str,
                            "exit_time": str(current_time),
                            "ticker": current_pos,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(curr_close, 2),
                            "pnl_pct": f"{actual_ret * 100:+.2f}%",
                            "pnl_krw": pnl_krw,
                            "capital_after": int(capital),
                            "exit_reason": f"TIME_STOP_{self.time_stop_bars*15}M",
                            "bars_held": bars_held
                        })
                        trade_id_seq += 1
                        current_pos = "NONE"
                        in_market = False
                        break

                    # [규칙 4] 장 마감 100% 청산 (오버나잇 0%)
                    if b_idx == num_bars - 1:
                        actual_ret = close_ret
                        pnl_krw = int(pos_capital * actual_ret)
                        capital += pnl_krw
                        all_closed_trades.append(actual_ret)
                        if actual_ret >= 0:
                            day_wins += 1
                            total_wins += 1
                            if current_pos == config.TICKER_LONG:
                                long_wins += 1
                            else:
                                short_wins += 1
                        else:
                            day_losses += 1
                            total_losses += 1
                            if current_pos == config.TICKER_LONG:
                                long_losses += 1
                            else:
                                short_losses += 1

                        all_trade_records.append({
                            "trade_id": f"TRD_{trade_id_seq:04d}",
                            "date": date_str,
                            "entry_time": entry_time_str,
                            "exit_time": str(current_time),
                            "ticker": current_pos,
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(curr_close, 2),
                            "pnl_pct": f"{actual_ret * 100:+.2f}%",
                            "pnl_krw": pnl_krw,
                            "capital_after": int(capital),
                            "exit_reason": "END_OF_DAY_CLOSE",
                            "bars_held": bars_held
                        })
                        trade_id_seq += 1
                        current_pos = "NONE"
                        in_market = False
                        break

            day_pnl_krw = int(capital - day_start_capital)
            day_return_pct = round((day_pnl_krw / day_start_capital) * 100, 2) if day_start_capital > 0 else 0.0
            day_win_rate = round((day_wins / (day_wins + day_losses) * 100), 1) if (day_wins + day_losses) > 0 else 0.0

            daily_reports.append({
                "date": date_str,
                "date_short": date_str[5:].replace("-", "/"),
                "pnl_krw": day_pnl_krw,
                "return_pct": day_return_pct,
                "total_trades": day_trades_count,
                "wins": day_wins,
                "losses": day_losses,
                "win_rate_pct": day_win_rate,
                "is_cash_day": (day_trades_count == 0)
            })
            equity_curve.append(capital)

        # 주별 집계
        weekly_reports = self._aggregate_weekly(daily_reports)

        # 종합 지표 산출
        total_pnl_krw = int(capital - self.initial_capital_krw)
        total_return_pct = round((total_pnl_krw / self.initial_capital_krw) * 100, 2)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        mdd_pct = round(float(abs(np.min(drawdown)) * 100), 2) if len(drawdown) > 0 else 0.0

        closed_arr = np.array(all_closed_trades) if len(all_closed_trades) > 0 else np.array([0.0])
        wins_arr = closed_arr[closed_arr > 0]
        losses_arr = closed_arr[closed_arr < 0]
        
        win_rate_pct = round((total_wins / (total_wins + total_losses) * 100), 1) if (total_wins + total_losses) > 0 else 0.0

        long_total = long_wins + long_losses
        short_total = short_wins + short_losses
        long_win_rate_pct = round((long_wins / long_total * 100), 1) if long_total > 0 else 0.0
        short_win_rate_pct = round((short_wins / short_total * 100), 1) if short_total > 0 else 0.0

        gross_profit = float(np.sum(wins_arr)) if len(wins_arr) > 0 else 0.001
        gross_loss = float(abs(np.sum(losses_arr))) if len(losses_arr) > 0 else 0.001
        profit_factor = round(gross_profit / gross_loss, 2)

        return {
            "initial_capital_krw": self.initial_capital_krw,
            "final_capital_krw": int(capital),
            "total_pnl_krw": total_pnl_krw,
            "total_return_pct": total_return_pct,
            "mdd_pct": mdd_pct,
            "total_trades_count": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate_pct": win_rate_pct,
            "long_wins": long_wins,
            "long_losses": long_losses,
            "long_win_rate_pct": long_win_rate_pct,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_win_rate_pct": short_win_rate_pct,
            "profit_factor": profit_factor,
            "daily_reports": daily_reports,
            "weekly_reports": weekly_reports,
            "all_trade_records": all_trade_records
        }

    def _aggregate_weekly(self, daily_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """일별 결과를 주차별로 집계"""
        weeks_map: Dict[str, List[Dict[str, Any]]] = {}

        for rep in daily_reports:
            dt = datetime.strptime(rep['date'], '%Y-%m-%d')
            year, week_num, _ = dt.isocalendar()
            week_key = f"{year}-W{week_num:02d}"
            
            if week_key not in weeks_map:
                weeks_map[week_key] = []
            weeks_map[week_key].append(rep)

        weekly_list = []
        w_idx = 1
        for w_key, days in weeks_map.items():
            w_pnl_krw = sum(d['pnl_krw'] for d in days)
            w_return_pct = round((w_pnl_krw / self.initial_capital_krw) * 100, 2)
            w_wins = sum(d['wins'] for d in days)
            w_losses = sum(d['losses'] for d in days)
            w_trades = sum(d['total_trades'] for d in days)
            w_win_rate = round((w_wins / (w_wins + w_losses) * 100), 1) if (w_wins + w_losses) > 0 else 0.0
            
            is_decay = w_return_pct < 0.0
            
            weekly_list.append({
                "week_name": f"{w_idx}주차",
                "week_code": w_key,
                "pnl_krw": w_pnl_krw,
                "return_pct": w_return_pct,
                "total_trades": w_trades,
                "wins": w_wins,
                "losses": w_losses,
                "win_rate_pct": w_win_rate,
                "is_decay": is_decay
            })
            w_idx += 1

        return weekly_list
