import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def run_comparison_backtest():
    print("1. 데이터 다운로드 (2010년부터 현재까지)")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df_wiki = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df_wiki['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    start_date = '2010-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading data...")
    data_unadj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, threads=True)
    data_adj = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, threads=True)
    
    print("2. 데이터 전처리")
    prices_u = data_unadj['Close'].ffill()
    
    data_actions = yf.download(tickers, start=start_date, end=end_date, actions=True, threads=True)
    if 'Dividends' in data_actions.columns:
        divs = data_actions['Dividends'].fillna(0)
    else:
        divs = pd.DataFrame(0, index=prices_u.index, columns=prices_u.columns)
        
    prices_a = data_adj['Close'].ffill()
    volumes = data_adj['Volume'].ffill()
    
    prices_u.index = prices_u.index.tz_localize(None)
    divs.index = divs.index.tz_localize(None)
    prices_a.index = prices_a.index.tz_localize(None)
    volumes.index = volumes.index.tz_localize(None)
    
    # 20일 평균 거래대금 계산 (시가총액 대용치 및 유니버스 필터링용)
    dollar_volume = prices_a * volumes
    recent_dollar_vol = dollar_volume.iloc[-20:].mean()
    universe = recent_dollar_vol.sort_values(ascending=False).index.tolist()[:350]
    
    prices_u = prices_u[universe]
    divs = divs[universe]
    prices_a = prices_a[universe]
    dollar_volume = dollar_volume[universe]
    
    # 시가총액 프록시 (20일 평균 거래대금)
    roll_dollar_vol = dollar_volume.rolling(window=20, min_periods=1).mean()
    
    print("3. 일간 배당수익률 및 Z-Score 5년(1260일) 롤링 계산, 모멘텀 계산")
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    z_score = (div_yield - roll_mean) / roll_std
    
    # 6개월(126일) 모멘텀 수익률 계산
    mom_6m = prices_a.pct_change(periods=126)
    
    z_score = z_score.dropna(how='all')
    if len(z_score) == 0:
        return
        
    valid_dates = z_score.index
    
    initial_cash = 100000.0
    trading_fee = 0.001
    MAX_HOLDINGS = 15
    BLACKLIST = ['PYPL', 'NKE']
    
    print("4. 백테스트 진행 중...")
    
    lamos_portfolio, lamos_cash, lamos_values = {}, initial_cash, []
    dod_portfolio, dod_cash, dod_values = {}, initial_cash, []
    top10_portfolio, top10_cash, top10_values = {}, initial_cash, []
    mom_portfolio, mom_cash, mom_values = {}, initial_cash, []
    
    current_year_dod = -1
    current_year_top10 = -1
    current_month_mom = -1
    
    tp = 0.25
    sl = -0.15
    
    for date in valid_dates:
        today_p = prices_a.loc[date]
        
        # --- 1. Lamos ---
        today_z = z_score.loc[date].dropna()
        symbols_to_sell = []
        for sym, pos in lamos_portfolio.items():
            if pd.isna(today_p.get(sym)): continue
            ret = (today_p[sym] - pos['entry_price']) / pos['entry_price']
            if ret >= tp or ret <= sl:
                gross = pos['shares'] * today_p[sym]
                lamos_cash += (gross - gross * trading_fee)
                symbols_to_sell.append(sym)
        for sym in symbols_to_sell: del lamos_portfolio[sym]
            
        empty_slots = MAX_HOLDINGS - len(lamos_portfolio)
        today_z = today_z[today_z <= 10.0]
        today_z = today_z[~today_z.index.isin(BLACKLIST)].sort_values(ascending=False)
        
        if empty_slots > 0 and lamos_cash > 20:
            candidates = today_z.drop(index=list(lamos_portfolio.keys()), errors='ignore')
            buy_syms = candidates.nlargest(empty_slots).index
            buy_amount = lamos_cash / empty_slots
            for sym in buy_syms:
                price = today_p.get(sym)
                if pd.isna(price) or price <= 0:
                    empty_slots -= 1
                    continue
                shares = int(buy_amount // price)
                if shares > 0:
                    cost = shares * price
                    fee = cost * trading_fee
                    lamos_cash -= (cost + fee)
                    lamos_portfolio[sym] = {'shares': shares, 'entry_price': price, 'cost_basis': cost + fee}
                empty_slots -= 1
                
        val_lamos = lamos_cash
        for sym, pos in lamos_portfolio.items():
            if pd.isna(today_p.get(sym)): val_lamos += pos['cost_basis']
            else: val_lamos += pos['shares'] * today_p[sym]
        lamos_values.append(val_lamos)
        
        # --- 2. Dogs of S&P ---
        today_yield = div_yield.loc[date].dropna()
        if date.year != current_year_dod:
            for sym, pos in dod_portfolio.items():
                if pd.isna(today_p.get(sym)): continue
                gross = pos['shares'] * today_p[sym]
                dod_cash += (gross - gross * trading_fee)
            dod_portfolio.clear()
            
            buy_syms = today_yield.nlargest(MAX_HOLDINGS).index
            buy_amount = dod_cash / MAX_HOLDINGS
            for sym in buy_syms:
                price = today_p.get(sym)
                if pd.isna(price) or price <= 0: continue
                shares = int(buy_amount // price)
                if shares > 0:
                    cost = shares * price
                    dod_cash -= (cost + cost * trading_fee)
                    dod_portfolio[sym] = {'shares': shares, 'cost_basis': cost + cost*trading_fee}
            current_year_dod = date.year
            
        val_dod = dod_cash
        for sym, pos in dod_portfolio.items():
            if pd.isna(today_p.get(sym)): val_dod += pos['cost_basis']
            else: val_dod += pos['shares'] * today_p[sym]
        dod_values.append(val_dod)
        
        # --- 3. Top 10 S&P (Market Cap Proxy) ---
        today_vol = roll_dollar_vol.loc[date].dropna()
        if date.year != current_year_top10:
            for sym, pos in top10_portfolio.items():
                if pd.isna(today_p.get(sym)): continue
                gross = pos['shares'] * today_p[sym]
                top10_cash += (gross - gross * trading_fee)
            top10_portfolio.clear()
            
            buy_syms = today_vol.nlargest(10).index
            buy_amount = top10_cash / 10
            for sym in buy_syms:
                price = today_p.get(sym)
                if pd.isna(price) or price <= 0: continue
                shares = int(buy_amount // price)
                if shares > 0:
                    cost = shares * price
                    top10_cash -= (cost + cost * trading_fee)
                    top10_portfolio[sym] = {'shares': shares, 'cost_basis': cost + cost*trading_fee}
            current_year_top10 = date.year
            
        val_top10 = top10_cash
        for sym, pos in top10_portfolio.items():
            if pd.isna(today_p.get(sym)): val_top10 += pos['cost_basis']
            else: val_top10 += pos['shares'] * today_p[sym]
        top10_values.append(val_top10)
        
        # --- 4. 6-Month Momentum (Top 15, Monthly Rebalance) ---
        today_mom = mom_6m.loc[date].dropna()
        if date.month != current_month_mom:
            # Sell all
            for sym, pos in mom_portfolio.items():
                if pd.isna(today_p.get(sym)): continue
                gross = pos['shares'] * today_p[sym]
                mom_cash += (gross - gross * trading_fee)
            mom_portfolio.clear()
            
            # Buy top 15 by momentum
            buy_syms = today_mom.nlargest(MAX_HOLDINGS).index
            buy_amount = mom_cash / MAX_HOLDINGS
            for sym in buy_syms:
                price = today_p.get(sym)
                if pd.isna(price) or price <= 0: continue
                shares = int(buy_amount // price)
                if shares > 0:
                    cost = shares * price
                    mom_cash -= (cost + cost * trading_fee)
                    mom_portfolio[sym] = {'shares': shares, 'cost_basis': cost + cost*trading_fee}
            current_month_mom = date.month
            
        val_mom = mom_cash
        for sym, pos in mom_portfolio.items():
            if pd.isna(today_p.get(sym)): val_mom += pos['cost_basis']
            else: val_mom += pos['shares'] * today_p[sym]
        mom_values.append(val_mom)
        
    print("5. 벤치마크 및 차트 생성")
    bm_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)['Close']
    bm_data.index = bm_data.index.tz_localize(None)
    bm_ret = bm_data.pct_change()
    bm_ret = bm_ret[bm_ret.index >= valid_dates[0]].dropna()
    bm_cum_ret = (1 + bm_ret).cumprod() * initial_cash

    s_lamos = pd.Series(lamos_values, index=valid_dates)
    s_dod = pd.Series(dod_values, index=valid_dates)
    s_top10 = pd.Series(top10_values, index=valid_dates)
    s_mom = pd.Series(mom_values, index=valid_dates)
    
    plt.figure(figsize=(12, 6))
    plt.plot(s_top10.index, s_top10, label='Top 10 Mega Cap', color='green', linewidth=1.5)
    plt.plot(s_mom.index, s_mom, label='6-Mo Momentum (Top 15)', color='orange', linewidth=2)
    plt.plot(s_lamos.index, s_lamos, label='Lamos Strategy (Z-Score + TP/SL)', color='red', linewidth=2)
    plt.plot(s_dod.index, s_dod, label='Dogs of S&P (High Yield)', color='blue', linewidth=1.5)
    
    if not bm_cum_ret.empty:
        if isinstance(bm_cum_ret, pd.DataFrame): bm_plot = bm_cum_ret.iloc[:, 0]
        else: bm_plot = bm_cum_ret
        plt.plot(bm_plot.index, bm_plot, label='S&P 500', color='black', linestyle='--')
        
    plt.title('Lamos vs Momentum vs Top 10 vs Dogs of S&P')
    plt.ylabel('Portfolio Balance ($)')
    plt.xlabel('Date')
    plt.gca().get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend()
    plt.grid(True)
    
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\84db9762-7513-488e-8956-3a8fce2df4fc"
    os.makedirs(artifact_dir, exist_ok=True)
    chart_path = os.path.join(artifact_dir, 'lamos_vs_all.png')
    plt.savefig(chart_path)
    
    years = len(valid_dates) / 252
    cagr_lamos = (s_lamos.iloc[-1] / initial_cash) ** (1/years) - 1
    cagr_dod = (s_dod.iloc[-1] / initial_cash) ** (1/years) - 1
    cagr_top10 = (s_top10.iloc[-1] / initial_cash) ** (1/years) - 1
    cagr_mom = (s_mom.iloc[-1] / initial_cash) ** (1/years) - 1
    last_bm = float(bm_plot.iloc[-1])
    cagr_bm = (last_bm / initial_cash) ** (1/years) - 1
    
    print(f"\n[결과 요약]")
    print(f"Top 10 최종: ${s_top10.iloc[-1]:,.2f} (CAGR {cagr_top10*100:.2f}%)")
    print(f"6-Mo Momentum 최종: ${s_mom.iloc[-1]:,.2f} (CAGR {cagr_mom*100:.2f}%)")
    print(f"Lamos 최종: ${s_lamos.iloc[-1]:,.2f} (CAGR {cagr_lamos*100:.2f}%)")
    print(f"Dogs S&P 최종: ${s_dod.iloc[-1]:,.2f} (CAGR {cagr_dod*100:.2f}%)")
    print(f"S&P 500 최종: ${last_bm:,.2f} (CAGR {cagr_bm*100:.2f}%)")

if __name__ == "__main__":
    run_comparison_backtest()
