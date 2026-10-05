import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import FinanceDataReader as fdr
import os

def run_backtest():
    print("1. KOSPI 시가총액 상위 200 종목 가져오기 (KOSPI 200 대용)")
    df = fdr.StockListing('KOSPI')
    df = df.sort_values(by='Marcap', ascending=False)
    tickers = df['Code'].head(200).tolist()
    print(f"종목 수: {len(tickers)}")
    
    # 10년치 데이터 (5년은 Z-Score 계산용 웜업, 5년은 백테스트)
    start_date = (datetime.now() - relativedelta(years=15)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"2. {start_date} 부터 {end_date} 까지 데이터 수집")
    price_unadj_data = {}
    div_data = {}
    price_adj_data = {}
    
    # For speed, we can use history with space separated tickers
    # But yf.Tickers is good
    ks_tickers = [f"{t}.KS" for t in tickers]
    
    print("yfinance 다운로드 중... (unadjusted)")
    data_unadj = yf.download(ks_tickers, start=start_date, end=end_date, auto_adjust=False, actions=True, threads=True)
    
    print("yfinance 다운로드 중... (adjusted for returns)")
    data_adj = yf.download(ks_tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)
    
    # Process data
    prices_unadj = data_unadj['Close']
    divs = data_unadj.get('Dividends')
    if divs is None:
        divs = pd.DataFrame(0, index=prices_unadj.index, columns=prices_unadj.columns)
    prices_adj = data_adj['Close']
    
    prices_unadj.columns = [str(c).replace('.KS', '') for c in prices_unadj.columns]
    divs.columns = [str(c).replace('.KS', '') for c in divs.columns]
    prices_adj.columns = [str(c).replace('.KS', '') for c in prices_adj.columns]
    
    # tz_localize None
    prices_unadj.index = prices_unadj.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    prices_adj.index = prices_adj.index.tz_localize(None)
    
    print("3. Z-Score 계산")
    # TTM Dividends (rolling 252 days sum)
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_unadj
    
    # 5-year rolling mean and std (approx 1260 days)
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    
    z_score = (div_yield - roll_mean) / roll_std
    
    # Monthly resampling
    z_score_monthly = z_score.resample('BM').last()
    prices_adj_monthly = prices_adj.resample('BM').last()
    
    # Forward 1-month returns using Adjusted Close
    forward_returns = prices_adj_monthly.shift(-1) / prices_adj_monthly - 1
    
    top_n_list = [50, 100, 150, 200]
    results = {}
    
    print("4. 포트폴리오 백테스트")
    for n in top_n_list:
        portfolio_returns = []
        portfolio_dates = []
        
        for i in range(len(z_score_monthly) - 1):
            date = z_score_monthly.index[i]
            z_row = z_score_monthly.iloc[i].dropna()
            
            if len(z_row) < 100: # Need at least some valid data to start
                continue
                
            n_actual = min(n, len(z_row))
            top_n_tickers = z_row.nlargest(n_actual).index
            
            rets = forward_returns.iloc[i][top_n_tickers].dropna()
            if len(rets) > 0:
                port_ret = rets.mean()
                portfolio_returns.append(port_ret)
                portfolio_dates.append(z_score_monthly.index[i+1])
                
        ret_series = pd.Series(portfolio_returns, index=portfolio_dates)
        results[n] = ret_series
        
    print("5. 벤치마크 (KOSPI 200) 데이터 수집")
    bm_ticker = "^KS200"
    bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
    if bm_data.empty:
        # Fallback to KOSPI
        bm_ticker = "^KS11"
        bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
        
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    bm_monthly = bm_prices.resample('BM').last()
    bm_returns = bm_monthly.pct_change().shift(-1).dropna() # shift(-1) aligns with our forward returns logic?
    # Wait, simple pct_change is fine, we just need to align indices
    # Let's align by taking pct_change of monthly prices
    bm_returns = bm_monthly.pct_change()
    
    print("6. 결과 저장 및 차트 생성")
    plt.figure(figsize=(14, 8))
    
    # We will compute cumulative returns
    # Add 1 and cumprod
    
    # To align BM with portfolio, we start at the first date of portfolio
    start_test_date = results[50].index[0]
    
    for n in top_n_list:
        ret = results[n]
        ret = ret[ret.index >= start_test_date]
        cum_ret = (1 + ret).cumprod()
        plt.plot(cum_ret.index, cum_ret, label=f'Top {n}')
        
        # Save summary
        total_return = cum_ret.iloc[-1] - 1
        cagr = (1 + total_return) ** (252 / (len(ret) * 21)) - 1 # Roughly
        cagr = (cum_ret.iloc[-1]) ** (12 / len(ret)) - 1
        print(f"Top {n} - Total Return: {total_return*100:.2f}%, CAGR: {cagr*100:.2f}%")
        
    # Benchmark
    bm_ret = bm_returns[bm_returns.index >= start_test_date].dropna()
    if not bm_ret.empty:
        bm_cum_ret = (1 + bm_ret).cumprod()
        # Align lengths if needed
        # Just plot
        plt.plot(bm_cum_ret.index, bm_cum_ret, label='Benchmark (KOSPI 200)', color='black', linewidth=2, linestyle='--')
        
        total_return_bm = float(bm_cum_ret.iloc[-1].iloc[0]) - 1
        cagr_bm = float(bm_cum_ret.iloc[-1].iloc[0]) ** (12 / len(bm_ret)) - 1
        print(f"Benchmark - Total Return: {total_return_bm*100:.2f}%, CAGR: {cagr_bm*100:.2f}%")
        
    plt.title('Extreme Dividend Z-Score Strategy (KOSPI 200) - Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\a5114209-0e45-4488-ba3d-fade16f65f0c"
    plt.savefig(os.path.join(artifact_dir, 'kospi200_backtest.png'))
    print("Saved kospi200_backtest.png")
    
if __name__ == "__main__":
    run_backtest()
