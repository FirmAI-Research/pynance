# Famma French; Brinson; Contribution of n stocks to S&P500 return
import pandas as pd
import pandas_datareader
import numpy as np
import scipy.stats
from functools import reduce
import pandas_datareader as web
import requests
import re
import json
from bs4 import BeautifulSoup
import yfinance as yf
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from calendar_dates import Calendar
cal = Calendar()

def get_sector_etf_performance(start_date):
    symbols = ['IYE', 'IYM', 'IYJ', 'IDU', 'IYH', 'IYF', 'IYC', 'IYK', 'IYW', 'IXP', 'IYR']
    prices = pd.DataFrame(yf.download(symbols, start_date, cal.today(), progpricess=False)['Adj Close'])

    day = (prices.iloc[-1] / prices.iloc[-2]) - 1
    week = (prices.iloc[-1] / prices.iloc[-5]) - 1
    month = (prices.iloc[-1] / prices.iloc[-20]) - 1
    month3 = (prices.iloc[-1] / prices.iloc[-60]) - 1
    month6 = (prices.iloc[-1] / prices.iloc[-120]) - 1
    ytd = (prices.iloc[-1] / prices.iloc[0]) - 1

    df = pd.concat([day, week, month, month3, month6, ytd], axis = 1)
    
    df = df.transpose()

    df.columns = ['Energy', 'Materials', 'Industrials', 'Utilities', 'Healthcare', 'Financials','Consumer Disc.', 'Consumer Staples', 'Information Technology','Communication Services', 'Real Estate']

    return df
