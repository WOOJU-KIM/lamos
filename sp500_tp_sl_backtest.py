import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import os

def run_backtest():
    print("1. S&P 500 종목 가져오기")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    start_date = (datetime.now() - relativedelta(years=15)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("2. 데이터 다운로드 (unadjusted + dividends)")
    data_unadj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, actions=True, threads=True)
    
    print("3. 데이터 다운로드 (adjusted for returns)")
    data_adj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)
    
    prices_u = data_unadj['Close']
    divs = data_unadj.get('Dividends')
    if divs is None:
        divs = pd.DataFrame(0, index=prices_u.index, columns=prices_u.columns)
    prices_a = data_adj['Close']
    volumes = data_adj['Volume']
    
    prices_u.index = prices_u.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    prices_a.index = prices_a.index.tz_localize(None)
    volumes.index = volumes.index.tz_localize(None)
    
    # Sort by recent Dollar Volume
    recent_dollar_vol = (prices_a.iloc[-20:] * volumes.iloc[-20:]).mean()
    sorted_tickers = recent_dollar_vol.sort_values(ascending=False).index.tolist()
    
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    z_score = (div_yield - roll_mean) / roll_std
    
    z_score = z_score.dropna(how='all')
    valid_dates = z_score.index
    
    results = {}
    top_n_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    print("4. 백테스트 진행")
    for n in top_n_list:
        print(f"Testing Top {n} Universe...")
        universe = sorted_tickers[:n]
        z_univ = z_score[universe]
        p_univ = prices_a[universe]
        
        portfolio = {}
        cash = 10000.0
        daily_values = []
        
        for i, date in enumerate(valid_dates):
            today_z = z_univ.loc[date].dropna()
            today_p = p_univ.loc[date]
            
            cash_from_sales = 0
            slots_freed = 0
            symbols_to_sell = []
            
            for sym, pos in portfolio.items():
                if pd.isna(today_p.get(sym)):
                    continue
                current_price = today_p[sym]
                ret = (current_price - pos['entry_price']) / pos['entry_price']
                
                if ret >= 0.20 or ret <= -0.15:
                    cash_from_sales += pos['shares'] * current_price
                    slots_freed += 1
                    symbols_to_sell.append(sym)
                    
            for sym in symbols_to_sell:
                del portfolio[sym]
                
            if i == 0:
                slots_freed = 10
                cash_from_sales = cash
                cash = 0
                
            if slots_freed > 0:
                candidates = today_z.drop(index=list(portfolio.keys()), errors='ignore')
                buy_syms = candidates.nlargest(slots_freed).index
                
                if len(buy_syms) > 0:
                    cash_per_stock = cash_from_sales / len(buy_syms)
                    for sym in buy_syms:
                        buy_price = today_p[sym]
                        if pd.isna(buy_price) or buy_price <= 0:
                            cash += cash_per_stock
                            continue
                        portfolio[sym] = {
                            'shares': cash_per_stock / buy_price,
                            'entry_price': buy_price
                        }
                        cash_from_sales -= cash_per_stock
                    cash += cash_from_sales
                else:
                    cash += cash_from_sales
                    
            mtm = cash
            for sym, pos in portfolio.items():
                cp = today_p.get(sym, pos['entry_price'])
                if pd.isna(cp): cp = pos['entry_price']
                mtm += pos['shares'] * cp
            daily_values.append(mtm)
            
        results[n] = pd.Series(daily_values, index=valid_dates)
        
    print("5. 벤치마크 (S&P 500) 데이터 수집")
    bm_ticker = "^GSPC"
    bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    bm_ret = bm_prices.pct_change()
    start_test_date = valid_dates[0]
    bm_ret = bm_ret[bm_ret.index >= start_test_date].dropna()
    bm_cum_ret = (1 + bm_ret).cumprod() * 10000.0
    
    print("6. 결과 저장 및 차트 생성")
    plt.figure(figsize=(14, 8))
    
    for n in top_n_list:
        val = results[n]
        cum_ret = val / val.iloc[0]
        plt.plot(cum_ret.index, cum_ret, label=f'Top {n}')
        
        total_return = cum_ret.iloc[-1] - 1
        cagr = cum_ret.iloc[-1] ** (252 / len(val)) - 1
        print(f"Top {n} - Total Return: {total_return*100:.2f}%, CAGR: {cagr*100:.2f}%")
        
    if not bm_cum_ret.empty:
        bm_cum_ret_norm = bm_cum_ret / bm_cum_ret.iloc[0]
        plt.plot(bm_cum_ret_norm.index, bm_cum_ret_norm, label='Benchmark (S&P 500)', color='black', linewidth=2, linestyle='--')
        
        if isinstance(bm_cum_ret_norm.iloc[-1], pd.Series):
            last_val = bm_cum_ret_norm.iloc[-1].iloc[0]
        else:
            last_val = bm_cum_ret_norm.iloc[-1]
            
        total_return_bm = float(last_val) - 1
        cagr_bm = float(last_val) ** (252 / len(bm_cum_ret_norm)) - 1
        print(f"Benchmark - Total Return: {total_return_bm*100:.2f}%, CAGR: {cagr_bm*100:.2f}%")
        
    plt.title('S&P 500 Z-Score TP/SL Strategy (Portfolio 10 stocks, +20%/-15%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\a5114209-0e45-4488-ba3d-fade16f65f0c"
    plt.savefig(os.path.join(artifact_dir, 'sp500_doe_backtest.png'))
    print("Saved sp500_doe_backtest.png")
    
if __name__ == "__main__":
    run_backtest()
