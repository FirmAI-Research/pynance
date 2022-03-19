import pyfolio as pf 
import zipline

import yfinance as yf
from datetime import date

today = date.today()

returns = yf.download('AAPL','1975-01-01',today)['Adj Close'].resample('M').last().pct_change()
print(returns.head(2))
pf.create_returns_tear_sheet(returns, live_start_date='2015-12-1')
