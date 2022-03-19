import yfinance as yf # Import yahoo finance module
tesla = yf.Ticker("TSLA") # Passing Tesla Inc. ticker

opt = tesla.option_chain('2022-06-17') #retreiving option chains data for 17 June 2022
opt.calls
opt.puts
BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
c = mibian.BS([427.53, 300, 0.25, 4], volatility=60)
