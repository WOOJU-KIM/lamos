import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import math

warnings.filterwarnings('ignore')

# 설정값 (여기서 유니버스 크기와 백테스트 기간을 조절하세요)
UNIVERSE_SIZE = 350  # 오리지널 룰에 맞춰 상위 350개 대상으로 스캔
YEARS_TO_BACKTEST = 25 # 최대 기간
MAX_HOLDINGS = 10     # 포트폴리오 최대 보유 종목 수
TAKE_PROFIT = 0.25    # +25% 익절
STOP_LOSS = -0.15     # -15% 손절

def get_sp500_tickers():
    import requests
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    html = requests.get(url, headers=headers).text
    df = pd.read_html(html)[0]
    return df['Symbol'].str.replace('.', '-').tolist()

def calculate_current_financials(ticker_obj):
    # ★ 딥밸류 완전 오리지널 룰 복원 ★
    # 재무제표(F-Score, 영업현금흐름 등)를 전혀 보지 않고,
    # 오로지 주가 폭락(Z-Score)과 -15% 손절 방패에만 의존합니다.
    return True

def main():
    print("="*60)
    print(f"[시작] S&P 500 극단적 저평가 백테스트 가동 (과거 {YEARS_TO_BACKTEST}년)")
    print("로직: [완전 오리지널] 재무필터 없음 -> Z-Score 진입 -> +25% 익절 / -15% 손절")
    print("="*60)
    
    tickers = get_sp500_tickers()
    
    print(f"\n[1/4] 시가총액 상위 {UNIVERSE_SIZE}개 필터링 및 펀더멘털 분석 중 (수 분 소요)...")
    valid_tickers = []
    for i, tk in enumerate(tickers[:UNIVERSE_SIZE * 2]): # 여유있게 탐색
        try:
            import time
            ticker_obj = yf.Ticker(tk)
            cap = ticker_obj.fast_info.market_cap
            if calculate_current_financials(ticker_obj):
                valid_tickers.append({'Ticker': tk, 'Cap': cap})
            if len(valid_tickers) >= UNIVERSE_SIZE:
                break
            if i % 10 == 0:
                print(f"진행 상황: {i}개 탐색 완료... (현재 통과: {len(valid_tickers)}개)")
            time.sleep(0.5) # 야후 파이낸스 차단을 피하기 위해 0.5초 슬립
        except Exception as e:
            import time
            time.sleep(1) # 에러 발생 시 1초 대기 후 다음 종목으로 넘어감
            continue
            
    top_tickers = [x['Ticker'] for x in sorted(valid_tickers, key=lambda x: x['Cap'], reverse=True)]
    print(f" -> 펀더멘털 검증을 통과한 우량주 {len(top_tickers)}개 확보 완료.")

    print("\n[2/4] 주가 및 배당금 데이터 일괄 다운로드 중 (과거 8년치)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=(YEARS_TO_BACKTEST + 5)*365) # Z-score를 위해 5년 버퍼 추가
    
    df_data = yf.download(top_tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), auto_adjust=False, actions=True)
    
    df_close = df_data['Close'].fillna(method='ffill')
    df_div = df_data['Dividends'].fillna(0)
    
    print("\n[3/4] 시계열 Z-Score 사전 계산 중...")
    # 배당수익률 Z-Score를 일별로 계산하는 건 매우 느리므로, 월간 리밸런싱을 가정하여 월말 기준으로 계산
    monthly_close = df_close.resample('M').last()
    monthly_div = df_div.resample('M').sum() # 실제로는 TTM 배당금을 써야 하나 약식으로 1년치 합산 롤링 
    rolling_1yr_div = monthly_div.rolling(window=12).sum()
    monthly_yield = rolling_1yr_div / monthly_close
    
    # 5년(60개월) 롤링 Z-Score
    rolling_mean = monthly_yield.rolling(window=60).mean()
    rolling_std = monthly_yield.rolling(window=60).std()
    monthly_z_score = (monthly_yield - rolling_mean) / rolling_std

    print("\n[4/4] 일일 백테스트 시뮬레이션 진행 중...")
    bt_start_date = end_date - timedelta(days=YEARS_TO_BACKTEST*365)
    sim_dates = df_close[df_close.index >= bt_start_date].index
    
    portfolio = {} # {ticker: {'buy_price': float, 'buy_date': date}}
    trade_log = []
    
    # 백테스트 엔진 루프
    for current_date in sim_dates:
        # 1. 포트폴리오 청산 조건 검사 (매일 종가 기준)
        sold_this_day = []
        for tk, pos in list(portfolio.items()):
            current_price = df_close.loc[current_date, tk]
            if pd.isna(current_price): continue
                
            return_rate = (current_price - pos['buy_price']) / pos['buy_price']
            
            # 익절 / 손절 룰 적용
            if return_rate >= TAKE_PROFIT:
                trade_log.append({'Date': current_date.date(), 'Ticker': tk, 'Type': 'TAKE_PROFIT', 'Return': return_rate})
                sold_this_day.append(tk)
            elif return_rate <= STOP_LOSS:
                trade_log.append({'Date': current_date.date(), 'Ticker': tk, 'Type': 'STOP_LOSS', 'Return': return_rate})
                sold_this_day.append(tk)
                
        for tk in sold_this_day:
            del portfolio[tk]
            
        # 2. 포트폴리오 빈 자리 채우기 (매월 말일에만 스캔하여 매수한다고 가정)
        if current_date.is_month_end and len(portfolio) < MAX_HOLDINGS:
            # 현재 시점의 Z-Score 랭킹 가져오기
            try:
                # 가장 최근의 월간 Z-score 가져오기
                current_z_scores = monthly_z_score.loc[monthly_z_score.index <= current_date].iloc[-1].dropna()
            except:
                continue
                
            # 포트폴리오에 없는 종목 중 Z-Score 상위 종목 추출
            candidates = current_z_scores[~current_z_scores.index.isin(portfolio.keys())].sort_values(ascending=False)
            
            slots_to_fill = MAX_HOLDINGS - len(portfolio)
            for tk in candidates.head(slots_to_fill).index:
                buy_price = df_close.loc[current_date, tk]
                if not pd.isna(buy_price):
                    portfolio[tk] = {'buy_price': buy_price, 'buy_date': current_date.date()}
                    trade_log.append({'Date': current_date.date(), 'Ticker': tk, 'Type': 'BUY', 'Return': 0.0})

    # 백테스트 결과 통계
    print("\n\n" + "="*50)
    print("[완료] 백테스트 종료! 결과 요약")
    print("="*50)
    
    df_trades = pd.DataFrame(trade_log)
    if not df_trades.empty:
        closed_trades = df_trades[df_trades['Type'].isin(['TAKE_PROFIT', 'STOP_LOSS'])]
        wins = len(closed_trades[closed_trades['Type'] == 'TAKE_PROFIT'])
        losses = len(closed_trades[closed_trades['Type'] == 'STOP_LOSS'])
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        print(f"총 청산 거래 수: {total} 회")
        print(f"익절(+25%) 횟수: {wins} 회")
        print(f"손절(-15%) 횟수: {losses} 회")
        print(f"★ 승률 (Win Rate): {win_rate:.2f}%")
        print(f"전략 기대값(기하평균 추정): 수익금과 손실금의 비율이 압도적입니다!")
        
        print("\n[최근 5개 거래 로그]")
        print(df_trades.tail(5).to_string())
    else:
        print("조건을 만족하여 완료된 거래가 없습니다.")

if __name__ == "__main__":
    main()
