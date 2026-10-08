import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def run_holdings_doe():
    print("1. 데이터 준비 중...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    start_date = (datetime.now() - relativedelta(years=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data_unadj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, actions=True, threads=True)
    data_adj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)
    
    prices_u = data_unadj['Close'].ffill()
    divs = data_unadj.get('Dividends')
    if divs is None:
        divs = pd.DataFrame(0, index=prices_u.index, columns=prices_u.columns)
    else:
        divs = divs.fillna(0)
        
    prices_a = data_adj['Close'].ffill()
    volumes = data_adj['Volume'].ffill()
    
    prices_u.index = prices_u.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    prices_a.index = prices_a.index.tz_localize(None)
    volumes.index = volumes.index.tz_localize(None)
    
    recent_dollar_vol = (prices_a.iloc[-20:] * volumes.iloc[-20:]).mean()
    sorted_tickers = recent_dollar_vol.sort_values(ascending=False).index.tolist()
    
    universe = sorted_tickers[:350]
    prices_u = prices_u[universe]
    divs = divs[universe]
    prices_a = prices_a[universe]
    
    print("2. Z-Score 5년 롤링 계산 중...")
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    z_score = (div_yield - roll_mean) / roll_std
    
    z_score = z_score.dropna(how='all')
    valid_dates = z_score.index
    
    initial_cash = 63075.0
    tax_rate = 0.22
    exemption = 2500.0
    trading_fee = 0.001
    tp = 0.25
    sl = -0.15
    BLACKLIST = ['PYPL', 'NKE']
    
    holdings_list = [10, 15, 20, 25, 30]
    results = {}
    
    print("3. DOE 시뮬레이션 시작 (보유 종목 수: 10, 15, 20, 25, 30)")
    
    for MAX_HOLDINGS in holdings_list:
        print(f" -> Testing MAX_HOLDINGS = {MAX_HOLDINGS}...")
        portfolio = {}
        cash = initial_cash
        daily_values = []
        annual_realized_profit = 0.0
        current_year = valid_dates[0].year
        
        for i, date in enumerate(valid_dates):
            today_z = z_score.loc[date].dropna()
            today_p = prices_a.loc[date]
            
            if date.year != current_year:
                if annual_realized_profit > exemption:
                    tax = (annual_realized_profit - exemption) * tax_rate
                    cash -= tax
                annual_realized_profit = 0.0
                current_year = date.year
                
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
                    cash += net_proceeds
                    symbols_to_sell.append(sym)
                    
            for sym in symbols_to_sell:
                del portfolio[sym]
                
            empty_slots = MAX_HOLDINGS - len(portfolio)
            today_z = today_z[today_z <= 10.0]
            today_z = today_z[~today_z.index.isin(BLACKLIST)].sort_values(ascending=False)
            
            if empty_slots > 0 and cash > 20:
                candidates = today_z.drop(index=list(portfolio.keys()), errors='ignore')
                buy_syms = candidates.nlargest(empty_slots).index
                buy_amount = cash / empty_slots
                
                for sym in buy_syms:
                    price = today_p.get(sym)
                    if pd.isna(price) or price <= 0:
                        empty_slots -= 1
                        continue
                        
                    shares = int(buy_amount // price)
                    if shares > 0:
                        cost = shares * price
                        fee = cost * trading_fee
                        cash -= (cost + fee)
                        portfolio[sym] = {'shares': shares, 'entry_price': price, 'cost_basis': cost + fee}
                    empty_slots -= 1
                    
            if cash > 20:
                top_10 = today_z.head(10).index
                for sym in top_10:
                    if cash < 20:
                        break
                    if sym in portfolio:
                        price = today_p.get(sym)
                        if pd.isna(price) or price <= 0:
                            continue
                        extra_shares = int(cash // price)
                        if extra_shares > 0:
                            cost = extra_shares * price
                            fee = cost * trading_fee
                            if cash >= (cost + fee):
                                cash -= (cost + fee)
                                portfolio[sym]['shares'] += extra_shares
                                total_cost = portfolio[sym]['cost_basis'] + (cost + fee)
                                portfolio[sym]['cost_basis'] = total_cost
                                portfolio[sym]['entry_price'] = total_cost / portfolio[sym]['shares']
                                
            total_value = cash
            for sym, pos in portfolio.items():
                if pd.isna(today_p.get(sym)):
                    total_value += pos['cost_basis']
                else:
                    total_value += pos['shares'] * today_p[sym]
            daily_values.append(total_value)
            
        results[MAX_HOLDINGS] = pd.Series(daily_values, index=valid_dates)
        
    print("4. 벤치마크 계산 및 차트 저장")
    bm_ticker = "^GSPC"
    bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    bm_ret = bm_prices.pct_change()
    bm_ret = bm_ret[bm_ret.index >= valid_dates[0]].dropna()
    bm_cum_ret = (1 + bm_ret).cumprod() * initial_cash
    
    plt.figure(figsize=(14, 8))
    
    for mh in holdings_list:
        val = results[mh]
        cagr = (val.iloc[-1] / initial_cash) ** (252 / len(val)) - 1
        plt.plot(val.index, val, label=f'{mh} Stocks (CAGR: {cagr*100:.2f}%)')
        print(f"MAX_HOLDINGS = {mh:2d} -> Final: ${val.iloc[-1]:,.2f}, CAGR: {cagr*100:.2f}%")
        
    if not bm_cum_ret.empty:
        last_bm = float(bm_cum_ret.iloc[-1].iloc[0]) if isinstance(bm_cum_ret.iloc[-1], pd.Series) else float(bm_cum_ret.iloc[-1])
        bm_cagr = (last_bm / initial_cash) ** (252 / len(bm_cum_ret)) - 1
        plt.plot(bm_cum_ret.index, bm_cum_ret, label=f'S&P 500 (CAGR: {bm_cagr*100:.2f}%)', color='black', linestyle='--')
        print(f"S&P 500 Benchmark -> Final: ${last_bm:,.2f}, CAGR: {bm_cagr*100:.2f}%")
        
    plt.title('Lamos Quant: Number of Holdings DOE (10, 15, 20, 25, 30)')
    plt.xlabel('Date')
    plt.ylabel('Balance (USD)')
    plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\78f83d0a-1c94-471b-982a-ff3519075014"
    chart_path = os.path.join(artifact_dir, 'doe_holdings_chart.png')
    plt.savefig(chart_path)
    print("DONE")

if __name__ == "__main__":
    run_holdings_doe()
