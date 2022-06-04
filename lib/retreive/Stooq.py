from pathlib import Path
import requests
from io import BytesIO
from zipfile import ZipFile, BadZipFile

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.datasets import fetch_openml

class Stooq():
    pass


# @ todo

meta_data_path = '/Users/michaelsands/Desktop/pybalancer/data/wiki_prices.csv'

def get_stooq_prices_and_tickers(frequency='daily',
                                 market='us',
                                 asset_class='nasdaq etfs'):
    stooq_path = '/Users/michaelsands/Desktop/pybalancer/data/stooq_data/daily/us/nyse etfs/'

    print(stooq_path)

    path = Path(stooq_path)
    files = path.glob('**/*.txt')
    prices = []
 
    parse_dates = ['date']
    date_label = 'date'
    names = ['ticker', 'freq', 'date', 'time', 
             'open', 'high', 'low', 'close','volume', 'openint']

    usecols = ['ticker', 'open', 'high', 'low', 'close', 'volume']+ parse_dates
    for i, file in enumerate(files, 1):
        ticker = str(file.name)[:4]
        if i % 500 == 0:
            print(i)
        else:
            try:
                df = (pd.read_csv(
                    file,
                    header=0,
                    names = names,
                    usecols=usecols,
                    parse_dates=parse_dates))
                prices.append(df)

            except pd.errors.EmptyDataError:
                print('/tdata missing', file.stem)
                file.unlink()

    prices = (pd.concat(prices, ignore_index=True)
              .rename(columns=str.lower).set_index(['ticker', date_label])
              .apply(lambda x: pd.to_numeric(x, errors='coerce')))


    return prices

def init():
    prices = get_stooq_prices_and_tickers()
    print(prices)
    prices.info()

    prices = prices.unstack('ticker')
    print(prices)
    print(prices.index)

    prices.info()

    stocks = pd.read_csv(meta_data_path).loc[:, ['ticker', 'marketcap', 'ipoyear', 'sector']]
    for i in range(len(stocks['ticker'])):
        stocks['ticker'].iloc[i] = stocks['ticker'].iloc[i] + '.US'
    stocks = stocks.set_index('ticker')

    print(stocks)


    print(stocks.index)
    print(prices.columns)
    # intersection of stock idx & cols --> None :(, check original data & switch to nyse stocks dirs 1,2,3
    shared = prices.columns.intersection(stocks.index)
    stocks = stocks.loc[shared, :]
    stocks.info()
    prices = prices.loc[:, shared]
    prices.info()
    assert prices.shape[1] == stocks.shape[0]



    # monthly return time series

    monthly_prices = prices.resample('M').last()
    monthly_prices.info()

    print(monthly_prices)

    outlier_cutoff = 0.01
    data = pd.DataFrame()
    lags = [1, 2, 3, 6, 9, 12]
    for lag in lags:
        data[f'return_{lag}m'] = (monthly_prices
                               .pct_change(lag)
                               .stack()
                               .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                      upper=x.quantile(1-outlier_cutoff)))
                               .add(1)
                               .pow(1/lag)
                               .sub(1)
                               )
    data.info()
    print(data)
    data = data.swaplevel().dropna()


if __name__ == '__main__':
    init()