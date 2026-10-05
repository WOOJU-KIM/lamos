import yfinance as yf
import pandas as pd
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# disable ssl verification for wikipedia
ssl._create_default_https_context = ssl._create_unverified_context

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
tables = pd.read_html(url)
sp500_table = tables[0]
tickers = sp500_table['Symbol'].tolist()
tickers = [t.replace('.', '-') for t in tickers] # BRK.B -> BRK-B

def get_mcap(t):
    try:
        return t, yf.Ticker(t).fast_info['marketCap']
    except:
        return t, 0

mcaps = {}
with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(get_mcap, tickers[:50]) # test first 50

for r in results:
    mcaps[r[0]] = r[1]
    
print(list(mcaps.items())[:5])
