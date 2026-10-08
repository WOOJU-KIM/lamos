import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def run_balance_backtest():
    print("1. S&P 500 종목 리스트 가져오기 및 데이터 다운로드 (고속)")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    # 25년치 데이터
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=25 + 5) # Z-score 5년 버퍼
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print("데이터 다운로드 중...")
    data_unadj = yf.download(tickers, start=start_str, end=end_str, auto_adjust=False, actions=True, threads=True)
    prices_u = data_unadj['Close']
    divs = data_unadj.get('Dividends')
    if divs is None:
        divs = pd.DataFrame(0, index=prices_u.index, columns=prices_u.columns)
        
    prices_u.index = prices_u.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    
    # 거래량 기반 유니버스 350개 선정
    data_adj = yf.download(tickers, start=start_str, end=end_str, auto_adjust=True, threads=True)
    prices_a = data_adj['Close']
    volumes = data_adj['Volume']
    prices_a.index = prices_a.index.tz_localize(None)
    volumes.index = volumes.index.tz_localize(None)
    
    recent_dollar_vol = (prices_a.iloc[-20:] * volumes.iloc[-20:]).mean()
    sorted_tickers = recent_dollar_vol.sort_values(ascending=False).index.tolist()
    universe = sorted_tickers[:350]
    
    prices = prices_u[universe].fillna(method='ffill')
    divs = divs[universe].fillna(0)
    
    print("Z-Score 계산 중...")
    monthly_close = prices.resample('M').last()
    monthly_div = divs.resample('M').sum()
    rolling_1yr_div = monthly_div.rolling(window=12).sum()
    monthly_yield = rolling_1yr_div / monthly_close
    
    rolling_mean = monthly_yield.rolling(window=60).mean()
    rolling_std = monthly_yield.rolling(window=60).std()
    monthly_z_score = (monthly_yield - rolling_mean) / rolling_std
    
    print("백테스트 시뮬레이션 중...")
    bt_start_date = end_date - relativedelta(years=25)
    sim_dates = prices[prices.index >= bt_start_date].index
    
    INITIAL_CASH = 20000000.0 # 2천만원
    MAX_HOLDINGS = 10
    TAKE_PROFIT = 0.25
    STOP_LOSS = -0.15
    
    cash = INITIAL_CASH
    portfolio = {} # ticker -> {'buy_price': float, 'shares': float}
    
    daily_values = []
    
    for i, current_date in enumerate(sim_dates):
        today_p = prices.loc[current_date]
        
        # 1. 청산 조건 검사 (매일)
        sold_this_day = []
        for tk, pos in list(portfolio.items()):
            current_price = today_p.get(tk)
            if pd.isna(current_price): continue
            
            return_rate = (current_price - pos['buy_price']) / pos['buy_price']
            
            if return_rate >= TAKE_PROFIT or return_rate <= STOP_LOSS:
                cash += pos['shares'] * current_price
                sold_this_day.append(tk)
                
        for tk in sold_this_day:
            del portfolio[tk]
            
        # 2. 매수 (월말)
        if current_date.is_month_end and len(portfolio) < MAX_HOLDINGS:
            try:
                current_z_scores = monthly_z_score.loc[monthly_z_score.index <= current_date].iloc[-1].dropna()
                candidates = current_z_scores[~current_z_scores.index.isin(portfolio.keys())].sort_values(ascending=False)
            except:
                candidates = pd.Series()
                
            slots_to_fill = MAX_HOLDINGS - len(portfolio)
            if slots_to_fill > 0 and len(candidates) > 0:
                buy_syms = candidates.head(slots_to_fill).index
                cash_per_stock = cash / slots_to_fill
                
                for tk in buy_syms:
                    buy_price = today_p.get(tk)
                    if pd.notna(buy_price) and buy_price > 0:
                        shares = cash_per_stock / buy_price
                        portfolio[tk] = {'buy_price': buy_price, 'shares': shares}
                        cash -= cash_per_stock
                        
        # 3. 일별 잔고 계산
        mtm = cash
        for tk, pos in portfolio.items():
            cp = today_p.get(tk)
            if pd.isna(cp): cp = pos['buy_price']
            mtm += pos['shares'] * cp
            
        daily_values.append(mtm)
        
    val_series = pd.Series(daily_values, index=sim_dates)
    
    # 벤치마크 (S&P 500)
    bm_ticker = "^GSPC"
    bm_data = yf.download(bm_ticker, start=bt_start_date.strftime('%Y-%m-%d'), end=end_str, auto_adjust=True)
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    
    # 리인덱싱하여 길이 맞춤
    bm_prices = bm_prices.reindex(sim_dates, method='ffill')
    bm_cum_ret = (bm_prices / bm_prices.iloc[0]) * INITIAL_CASH
    
    final_balance = val_series.iloc[-1]
    total_return = (final_balance / INITIAL_CASH) - 1
    years = (sim_dates[-1] - sim_dates[0]).days / 365.25
    cagr = ((final_balance / INITIAL_CASH) ** (1 / years)) - 1
    
    print(f"\n최종 잔액: {final_balance:,.0f} 원")
    print(f"누적 수익률: {total_return*100:.2f}%")
    print(f"연평균 수익률(CAGR): {cagr*100:.2f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(val_series.index, val_series, label=f'Strategy (TP +25%, SL -15%) - CAGR {cagr*100:.2f}%', color='blue')
    plt.plot(bm_cum_ret.index, bm_cum_ret, label='S&P 500 Benchmark', color='gray', linestyle='--')
    plt.title('Equity Curve (20M KRW Initial Balance)')
    plt.xlabel('Date')
    plt.ylabel('Balance (KRW)')
    
    plt.gca().get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\78f83d0a-1c94-471b-982a-ff3519075014"
    os.makedirs(artifact_dir, exist_ok=True)
    chart_path = os.path.join(artifact_dir, 'balance_chart.png')
    plt.savefig(chart_path)
    print(f"차트 저장 완료: {chart_path}")

if __name__ == "__main__":
    run_balance_backtest()
