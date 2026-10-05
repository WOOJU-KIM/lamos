import yfinance as yf
import pandas as pd

ticker = yf.Ticker("005930.KS")
hist = ticker.history(period="10y", auto_adjust=False)
divs = ticker.dividends

print("History tail:\n", hist.tail())
print("Dividends tail:\n", divs.tail())
