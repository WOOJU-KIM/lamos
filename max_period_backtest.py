import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# ===============================================
# 절대 룰 (GEMINI.md) 적용 백테스트 스크립트
# ===============================================

def run_max_backtest():
    print("1. S&P 500 종목 데이터 다운로드 (최대 기간 30년치)")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    df = pd.read_html(url, storage_options={'User-Agent': 'Mozilla/5.0'})[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    
    # 30년치 데이터를 받아야 최소 25년치 백테스트가 가능 (5년치 Z-Score 계산 버퍼)
    start_date = (datetime.now() - relativedelta(years=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 배당 계산용 무보정(Unadjusted) 데이터 & 현재가액/거래대금 계산용 수정(Adjusted) 데이터
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
    
    print("2. 거래대금 상위 350개 유니버스 필터링 (최근 20일 기준)")
    recent_dollar_vol = (prices_a.iloc[-20:] * volumes.iloc[-20:]).mean()
    sorted_tickers = recent_dollar_vol.sort_values(ascending=False).index.tolist()
    
    universe = sorted_tickers[:350]
    prices_u = prices_u[universe]
    divs = divs[universe]
    prices_a = prices_a[universe]
    
    print("3. 일간 배당수익률 및 Z-Score 5년(1260일) 롤링 계산")
    ttm_divs = divs.rolling(window=252, min_periods=1).sum()
    div_yield = ttm_divs / prices_u
    
    roll_mean = div_yield.rolling(window=1260, min_periods=1260).mean()
    roll_std = div_yield.rolling(window=1260, min_periods=1260).std()
    z_score = (div_yield - roll_mean) / roll_std
    
    z_score = z_score.dropna(how='all')
    valid_dates = z_score.index
    
    print("4. 백테스트 엔진 가동 (최대 보유 20종목, 잔돈 싹쓸이 기능 탑재)")
    initial_cash = 63075.0 # USD 기준 현재 잔고와 유사하게 설정
    tax_rate = 0.22 # 양도소득세 22%
    exemption = 2500.0 # 2500달러 비과세 공제
    trading_fee = 0.001 # 수수료 0.1%
    
    tp = 0.25 # 익절 25%
    sl = -0.15 # 손절 -15%
    MAX_HOLDINGS = 20
    BLACKLIST = ['PYPL', 'NKE']
    
    portfolio = {}
    cash = initial_cash
    daily_values = []
    
    annual_realized_profit = 0.0
    current_year = valid_dates[0].year
    total_tax_paid = 0.0
    
    for i, date in enumerate(valid_dates):
        today_z = z_score.loc[date].dropna()
        today_p = prices_a.loc[date]
        
        # 연초 세금 정산
        if date.year != current_year:
            if annual_realized_profit > exemption:
                tax = (annual_realized_profit - exemption) * tax_rate
                cash -= tax
                total_tax_paid += tax
            annual_realized_profit = 0.0
            current_year = date.year
            
        # 1. 매도 (익절 / 손절)
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
            
        # 2. 신규 매수 (1차 N빵 + 2차 짤짤이 몰빵)
        empty_slots = MAX_HOLDINGS - len(portfolio)
        
        # 필터링: Z-Score 10 초과 제외 및 블랙리스트 제외
        today_z = today_z[today_z <= 10.0]
        today_z = today_z[~today_z.index.isin(BLACKLIST)].sort_values(ascending=False)
        
        if empty_slots > 0 and cash > 20:
            candidates = today_z.drop(index=list(portfolio.keys()), errors='ignore')
            buy_syms = candidates.nlargest(empty_slots).index
            
            # 1차 N빵
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
                    
                    portfolio[sym] = {
                        'shares': shares,
                        'entry_price': price,
                        'cost_basis': cost + fee
                    }
                empty_slots -= 1
                
        # 2차 매수 (잔돈 털기) - Z-Score Top 10 순서대로 남은 돈 싹쓸이
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
                        
                        # 수수료 포함해도 돈이 충분한지 체크
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            portfolio[sym]['shares'] += extra_shares
                            
                            # 평단가 리밸런싱
                            total_cost = portfolio[sym]['cost_basis'] + (cost + fee)
                            portfolio[sym]['cost_basis'] = total_cost
                            portfolio[sym]['entry_price'] = total_cost / portfolio[sym]['shares']
                            
        # 일일 평가액 기록
        total_value = cash
        for sym, pos in portfolio.items():
            if pd.isna(today_p.get(sym)):
                total_value += pos['cost_basis'] # 상폐나 정지시 원금으로 가정
            else:
                total_value += pos['shares'] * today_p[sym]
                
        daily_values.append(total_value)
        
    val_series = pd.Series(daily_values, index=valid_dates)
    
    print("5. 벤치마크(S&P 500) 비교 및 차트 생성")
    bm_ticker = "^GSPC"
    bm_data = yf.download(bm_ticker, start=start_date, end=end_date, auto_adjust=True)
    bm_prices = bm_data['Close']
    bm_prices.index = bm_prices.index.tz_localize(None)
    bm_ret = bm_prices.pct_change()
    bm_ret = bm_ret[bm_ret.index >= valid_dates[0]].dropna()
    bm_cum_ret = (1 + bm_ret).cumprod() * initial_cash
    
    plt.figure(figsize=(14, 8))
    plt.plot(val_series.index, val_series, label='Lamos Extreme Div Quant (20 Stocks, Cash Sweep)', color='blue')
    
    if not bm_cum_ret.empty:
        plt.plot(bm_cum_ret.index, bm_cum_ret, label='S&P 500 Benchmark', color='black', linestyle='--')
        
    plt.title('Lamos Strategy Backtest (Max Period, Tax/Fee Included)')
    plt.xlabel('Date')
    plt.ylabel('Balance (USD)')
    
    plt.gca().get_yaxis().set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    artifact_dir = r"C:\Users\chabo\.gemini\antigravity\brain\78f83d0a-1c94-471b-982a-ff3519075014"
    chart_path = os.path.join(artifact_dir, 'lamos_max_backtest_chart.png')
    plt.savefig(chart_path)
    
    cagr = (val_series.iloc[-1] / initial_cash) ** (252 / len(val_series)) - 1
    bm_cagr = (float(bm_cum_ret.iloc[-1].iloc[0]) if isinstance(bm_cum_ret.iloc[-1], pd.Series) else float(bm_cum_ret.iloc[-1])) / initial_cash ** (252 / len(bm_cum_ret)) - 1 if not bm_cum_ret.empty else 0
    
    # Fix bm_cagr formula
    last_bm = float(bm_cum_ret.iloc[-1].iloc[0]) if isinstance(bm_cum_ret.iloc[-1], pd.Series) else float(bm_cum_ret.iloc[-1])
    bm_cagr = (last_bm / initial_cash) ** (252 / len(bm_cum_ret)) - 1
    
    print(f"\n[백테스트 결과 요약]")
    print(f"- 전략 최종 자산: ${val_series.iloc[-1]:,.2f} (CAGR: {cagr*100:.2f}%)")
    print(f"- S&P 500 자산:  ${last_bm:,.2f} (CAGR: {bm_cagr*100:.2f}%)")
    print(f"- 납부 세금 총액: ${total_tax_paid:,.2f}")
    
if __name__ == "__main__":
    run_max_backtest()
