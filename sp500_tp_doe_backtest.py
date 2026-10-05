import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def run_tp_doe_backtest():
    print("1. S&P 500 종목 및 데이터 준비")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    start_date = (datetime.now() - relativedelta(years=15)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data_unadj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, actions=True, threads=True)
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
    
    recent_dollar_vol = (prices_a.iloc[-20:] * volumes.iloc[-20:]).mean()
    sorted_tickers = recent_dollar_vol.sort_values(ascending=False).index.tolist()
    
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    z_score = (div_yield - roll_mean) / roll_std
    
    z_score = z_score.dropna(how='all')
    valid_dates = z_score.index
    
    # ---------------------------------------------------------
    # 고정 설정: Top 350
    n = 350
    universe = sorted_tickers[:n]
    z_univ = z_score[universe]
    p_univ = prices_a[universe]
    
    initial_cash = 100000.0
    exchange_rate = 1300.0
    exemption_usd = 2500000 / exchange_rate
    tax_rate = 0.22
    trading_fee = 0.001
    
    tp_list = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sl = -0.15
    results = {}
    
    for tp in tp_list:
        print(f"Testing TP: +{tp*100:.0f}%, SL: {sl*100:.0f}%")
        portfolio = {}
        cash = initial_cash
        daily_values = []
        annual_realized_profit = 0.0
        current_year = valid_dates[0].year
        total_tax_paid = 0.0
        total_fee_paid = 0.0
        
        for i, date in enumerate(valid_dates):
            today_z = z_univ.loc[date].dropna()
            today_p = p_univ.loc[date]
            
            if date.year != current_year:
                if annual_realized_profit > exemption_usd:
                    tax = (annual_realized_profit - exemption_usd) * tax_rate
                    cash -= tax
                    total_tax_paid += tax
                annual_realized_profit = 0.0
                current_year = date.year
                
            cash_from_sales = 0
            slots_freed = 0
            symbols_to_sell = []
            
            for sym, pos in portfolio.items():
                if pd.isna(today_p.get(sym)):
                    continue
                current_price = today_p[sym]
                ret = (current_price - pos['entry_price']) / pos['entry_price']
                
                if ret >= tp or ret <= sl:
                    gross_proceeds = pos['shares'] * current_price
                    fee = gross_proceeds * trading_fee
                    net_proceeds = gross_proceeds - fee
                    realized_pnl = net_proceeds - pos['cost_basis']
                    annual_realized_profit += realized_pnl
                    cash_from_sales += net_proceeds
                    total_fee_paid += fee
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
                        
                        fee = cash_per_stock * trading_fee
                        net_investment = cash_per_stock - fee
                        total_fee_paid += fee
                        
                        shares = net_investment / buy_price
                        portfolio[sym] = {
                            'shares': shares,
                            'entry_price': buy_price,
                            'cost_basis': net_investment
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
            
        val_series = pd.Series(daily_values, index=valid_dates)
        if annual_realized_profit > exemption_usd:
            tax = (annual_realized_profit - exemption_usd) * tax_rate
            total_tax_paid += tax
            val_series.iloc[-1] -= tax
            
        results[tp] = {
            'series': val_series,
            'tax': total_tax_paid,
            'fee': total_fee_paid
        }
        
    print("5. 벤치마크 데이터 수집")
    bm_ticker = "^GSPC"
    bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    bm_ret = bm_prices.pct_change()
    start_test_date = valid_dates[0]
    bm_ret = bm_ret[bm_ret.index >= start_test_date].dropna()
    bm_cum_ret = (1 + bm_ret).cumprod() * initial_cash
    
    print("6. 결과 출력 및 차트 생성")
    plt.figure(figsize=(14, 8))
    
    for tp in tp_list:
        res = results[tp]
        val = res['series']
        cum_ret = val / initial_cash
        plt.plot(cum_ret.index, cum_ret, label=f'TP +{tp*100:.0f}%')
        
        total_return = cum_ret.iloc[-1] - 1
        cagr = cum_ret.iloc[-1] ** (252 / len(val)) - 1
        print(f"TP +{tp*100:.0f}% -> Total Ret: {total_return*100:.2f}%, CAGR: {cagr*100:.2f}%, Tax: ${res['tax']:,.0f}")
        
    if not bm_cum_ret.empty:
        bm_cum_ret_norm = bm_cum_ret / initial_cash
        plt.plot(bm_cum_ret_norm.index, bm_cum_ret_norm, label='S&P 500', color='black', linewidth=2, linestyle='--')
        
        if isinstance(bm_cum_ret_norm.iloc[-1], pd.Series):
            last_val = bm_cum_ret_norm.iloc[-1].iloc[0]
        else:
            last_val = bm_cum_ret_norm.iloc[-1]
            
        total_return_bm = float(last_val) - 1
        cagr_bm = float(last_val) ** (252 / len(bm_cum_ret_norm)) - 1
        print(f"Benchmark -> Total Ret: {total_return_bm*100:.2f}%, CAGR: {cagr_bm*100:.2f}%")
        
    plt.title('Take Profit DOE (Top 350, SL -15%, With 22% Tax & Fees)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\a5114209-0e45-4488-ba3d-fade16f65f0c"
    plt.savefig(os.path.join(artifact_dir, 'sp500_tp_doe_backtest.png'))
    print("Saved sp500_tp_doe_backtest.png")

if __name__ == "__main__":
    run_tp_doe_backtest()
