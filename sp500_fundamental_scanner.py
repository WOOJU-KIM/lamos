import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def get_sp500_tickers():
    import requests
    print("1. S&P 500 종목 리스트 가져오는 중...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    html = requests.get(url, headers=headers).text
    tables = pd.read_html(html)
    df = tables[0]
    return df['Symbol'].str.replace('.', '-').tolist()

def calculate_f_score_and_revenue(ticker_obj):
    try:
        bs = ticker_obj.balance_sheet
        inc = ticker_obj.income_stmt
        cf = ticker_obj.cashflow
        
        # 재무제표 데이터가 충분하지 않으면 패스
        if bs.empty or inc.empty or cf.empty or len(bs.columns) < 2 or len(inc.columns) < 2:
            return 0, False
        
        cy = bs.columns[0] # Current Year (가장 최근 년도)
        py = bs.columns[1] # Prior Year (직전 년도)
        
        f_score = 0
        
        # 항목을 안전하게 가져오는 헬퍼 함수
        def safe_get(df, keywords, col):
            for idx in df.index:
                for kw in keywords:
                    if kw.lower() in idx.lower():
                        val = df.loc[idx, col]
                        if pd.isna(val): return 0
                        return val
            return 0

        # [1] 수익성 (Profitability)
        ni_cy = safe_get(inc, ['Net Income'], cy)
        ni_py = safe_get(inc, ['Net Income'], py)
        ta_cy = safe_get(bs, ['Total Assets'], cy)
        ta_py = safe_get(bs, ['Total Assets'], py)
        
        roa_cy = ni_cy / ta_cy if ta_cy else 0
        roa_py = ni_py / ta_py if ta_py else 0
        if roa_cy > 0: f_score += 1
        
        cfo_cy = safe_get(cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'], cy)
        if cfo_cy > 0: f_score += 1
        if roa_cy > roa_py: f_score += 1
        if cfo_cy > ni_cy: f_score += 1 # 이익 조작 방지 (Accruals)
        
        # [2] 재무 건전성 (Leverage & Liquidity)
        ld_cy = safe_get(bs, ['Long Term Debt'], cy)
        ld_py = safe_get(bs, ['Long Term Debt'], py)
        if (ld_cy/ta_cy if ta_cy else 0) < (ld_py/ta_py if ta_py else 0): f_score += 1
        
        ca_cy = safe_get(bs, ['Current Assets'], cy)
        cl_cy = safe_get(bs, ['Current Liabilities'], cy)
        ca_py = safe_get(bs, ['Current Assets'], py)
        cl_py = safe_get(bs, ['Current Liabilities'], py)
        cr_cy = ca_cy / cl_cy if cl_cy else 0
        cr_py = ca_py / cl_py if cl_py else 0
        if cr_cy > cr_py: f_score += 1
        
        shares_cy = safe_get(inc, ['Basic Average Shares', 'Ordinary Shares Number'], cy)
        shares_py = safe_get(inc, ['Basic Average Shares', 'Ordinary Shares Number'], py)
        if shares_cy <= shares_py: f_score += 1 # 유상증자 여부
        
        # [3] 효율성 (Operating Efficiency)
        rev_cy = safe_get(inc, ['Total Revenue', 'Operating Revenue'], cy)
        rev_py = safe_get(inc, ['Total Revenue', 'Operating Revenue'], py)
        gp_cy = safe_get(inc, ['Gross Profit'], cy)
        gp_py = safe_get(inc, ['Gross Profit'], py)
        
        gm_cy = gp_cy / rev_cy if rev_cy else 0
        gm_py = gp_py / rev_py if rev_py else 0
        if gm_cy > gm_py: f_score += 1
        
        at_cy = rev_cy / ta_cy if ta_cy else 0
        at_py = rev_py / ta_py if ta_py else 0
        if at_cy > at_py: f_score += 1
        
        # 켄 피셔 스타일: 매출 증가 여부 (Revenue Growth)
        rev_growth = rev_cy > rev_py
        
        return f_score, rev_growth
    except Exception as e:
        return 0, False

def get_dividend_z_score(ticker_obj):
    try:
        # 배당률 Z-Score를 구하기 위해 과거 5년 데이터 가져오기 (수정주가 미적용)
        hist = ticker_obj.history(period="5y", auto_adjust=False)
        if hist.empty or 'Dividends' not in hist.columns:
            return None, None
            
        # 간이 배당률 계산: 1년 단위 총 배당금 / 연말 종가
        divs = hist['Dividends'].resample('Y').sum()
        prices = hist['Close'].resample('Y').last()
        
        yields = divs / prices
        yields = yields.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(yields) < 3: return None, None # 데이터가 너무 적으면 패스
        
        current_yield = yields.iloc[-1]
        mean_yield = yields.mean()
        std_yield = yields.std()
        
        if std_yield == 0: return None, None
        z_score = (current_yield - mean_yield) / std_yield
        
        return z_score, current_yield
    except:
        return None, None

def main():
    tickers = get_sp500_tickers()
    
    # 2. 시가총액 정보 가져와서 상위 350개로 필터링 (간소화를 위해 yfinance info 활용)
    print("2. 시가총액 기준 S&P 500 상위 350개 유니버스 필터링 중...")
    cap_data = []
    for tk in tickers:
        try:
            info = yf.Ticker(tk).fast_info # fast_info가 훨씬 빠름
            cap_data.append({'Ticker': tk, 'MarketCap': info.market_cap})
        except:
            continue
            
    df_caps = pd.DataFrame(cap_data).sort_values('MarketCap', ascending=False)
    top_350_tickers = df_caps['Ticker'].head(350).tolist()
    
    results = []
    print(f"\n3. 총 {len(top_350_tickers)}개 종목 재무 스캔 시작! (수 분 소요 예정)")
    
    import time
    for i, tk in enumerate(top_350_tickers, 1):
        if i % 10 == 0:
            print(f"진행 상황: {i}/{len(top_350_tickers)}개 스캔 완료...")
        
        try:
            ticker_obj = yf.Ticker(tk)
            f_score, _ = calculate_f_score_and_revenue(ticker_obj)
        except Exception:
            time.sleep(1)
            continue
            
        time.sleep(0.5)
        
        # ★ 필터 조건: F-Score 6점 이상 (매출 성장 절대 조건 삭제) ★
        if f_score >= 6:
            # Z-Score 검사
            z_score, curr_yield = get_dividend_z_score(ticker_obj)
            
            # 배당이 존재하는 기업만
            if z_score is not None and curr_yield > 0:
                results.append({
                    'Ticker': tk,
                    'F_Score': f_score,
                    'Rev_Growth': 'Pass (증가)',
                    'Div_Yield(%)': round(curr_yield * 100, 2),
                    'Z_Score': round(z_score, 2)
                })
                
    if not results:
        print("\n\n조건을 만족하는 종목이 없습니다.")
        return
        
    df_results = pd.DataFrame(results)
    # Z-Score 기준 내림차순 정렬 (극단적 저평가 순)
    df_results = df_results.sort_values('Z_Score', ascending=False).reset_index(drop=True)
    
    print("\n\n" + "="*60)
    print("[완료] 스캔 완료: 가치함정을 피한 극단적 저평가 Top 10")
    print("="*60)
    print(df_results.head(10).to_string())

if __name__ == "__main__":
    main()
