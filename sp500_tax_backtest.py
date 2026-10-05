import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

def run_tax_backtest():
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
    
    # 설정
    n = 350
    universe = sorted_tickers[:n]
    z_univ = z_score[universe]
    p_univ = prices_a[universe]
    
    # ------------------ 택스 및 수수료 설정 ------------------
    # 초기 자본 10만 불 (약 1.3억 원)
    # 연 250만 원 기본공제 = 약 1923 달러
    initial_cash = 100000.0
    exchange_rate = 1300.0
    exemption_usd = 2500000 / exchange_rate
    tax_rate = 0.22
    trading_fee = 0.001 # 0.1% (매수/매도 시 각각 발생, 증권사 수수료+슬리피지 감안)
    # ---------------------------------------------------------
    
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
        
        # 해가 바뀌면 세금 정산 (지난 해 수익에 대해)
        if date.year != current_year:
            if annual_realized_profit > exemption_usd:
                tax = (annual_realized_profit - exemption_usd) * tax_rate
                cash -= tax
                total_tax_paid += tax
            # 초기화
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
            
            if ret >= 0.20 or ret <= -0.15:
                # 매도
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
                    
                    # 매수 수수료 차감
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
    
    # 마지막 해 세금 정산 (가정)
    if annual_realized_profit > exemption_usd:
        tax = (annual_realized_profit - exemption_usd) * tax_rate
        total_tax_paid += tax
        val_series.iloc[-1] -= tax
        
    print(f"--- Top 350 (세금+수수료 반영) 결과 ---")
    cum_ret = val_series / val_series.iloc[0]
    total_return = cum_ret.iloc[-1] - 1
    cagr = cum_ret.iloc[-1] ** (252 / len(val_series)) - 1
    print(f"Total Return: {total_return*100:.2f}%, CAGR: {cagr*100:.2f}%")
    print(f"Total Fee Paid: ${total_fee_paid:,.2f}")
    print(f"Total Tax Paid: ${total_tax_paid:,.2f}")
    print(f"Final Balance: ${val_series.iloc[-1]:,.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(cum_ret.index, cum_ret, label='Top 350 (After Tax & Fee)')
    plt.title('S&P 500 Top 350 TP/SL Strategy - REALISTIC (22% Tax, 0.1% Fee)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\a5114209-0e45-4488-ba3d-fade16f65f0c"
    plt.savefig(os.path.join(artifact_dir, 'sp500_realistic_backtest.png'))
    print("Saved sp500_realistic_backtest.png")
    
if __name__ == "__main__":
    run_tax_backtest()
