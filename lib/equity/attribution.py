# Famma French; Brinson; Contribution of n stocks to S&P500 return
import pandas as pd
import pandas_datareader
import numpy as np
import scipy.stats

import requests
import re
import json
from bs4 import BeautifulSoup
import yfinance as yf


class Attribution:

    def __init__(self):
        pass


    def get_holdings(self, ticker):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"
        }

        def main(url, tickers):
            with requests.Session() as req:
                req.headers.update(headers)    
                frames = []
                for key in tickers:
                    r = req.get(url.format(key))
                    print(f"Extracting: {r.url}")      
                    rows = []
                    for line in r.text.splitlines():
                        if not line.startswith('etf_holdings.formatted_data'):
                            continue
                        data = json.loads(line[30:-1])
                        for holding in data:
                            goal = re.search(r'etf/([^"]*)', holding[1])
                            if goal:
                                rows.append([goal.group(1), *holding[2:5]])
                    df = pd.DataFrame(rows, columns = ['Symbol', 'Shares', 'Weight', '52 week change'])
                    for err in ['NaN', 'NA', 'nan']:
                        df["Weight"] = df['Weight'].apply(lambda x : x.replace(err, ''))
                    
                    df.dropna(how='any', axis=0, inplace=True)
                    df = df[['Symbol', 'Shares', 'Weight']]
                    df['Shares'] = pd.to_numeric([x.replace(',','') for x in df['Shares']])
                    df['Weight'] = pd.to_numeric(df['Weight'])
                    df.dropna(how='any', axis=0, inplace=True)
                    frames.append(df)
            return frames
                    
        portfolio = main(url = "https://www.zacks.com/funds/etf/{}/holding", tickers = [ ticker ])
        print(portfolio)
        
        return portfolio