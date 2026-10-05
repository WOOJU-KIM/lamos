from pykrx import stock
# KOSPI 200 tickers
tickers = stock.get_index_portfolio_deposit_file("1028")
print(len(tickers))
print(tickers[:10])
