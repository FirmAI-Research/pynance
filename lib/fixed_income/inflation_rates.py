import sys, os
import numpy as np

from datetime import datetime
import pandas as pd
import numpy as np 
from re import L, X
import requests
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import json
import seaborn as sns
import yfinance as yf
from pathlib import Path
proj_root = Path(__file__).resolve().parent.parent.parent
print(proj_root)
sys.path.append(proj_root)
from calendar_dates import Calendar
import fredapi
import sys, os, json

from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols


fp = os.path.join(proj_root, 'secrets.json')
print(fp)
with open(fp) as f:
    data = json.load(f)
fred_api_key = data['fred_api_key'] 


import numeric
cal = Calendar()


def breakeven_inflation():
    fred = fredapi.Fred(api_key=fred_api_key)
    
    data = fred.get_series('T10YIE').to_frame().rename(columns={0:'10Y Breakeven'}).reset_index()
    
    data['index'] = data['index'].astype(str)
    
    data.set_index('index', inplace = True)
    
    data.dropna(axis=0, how = 'any', inplace = True)

    return data



def expected_inflation():

    # Moodys seasoned bond yields
    fred = fredapi.Fred(api_key=fred_api_key)
    data = fred.get_series('T20YIEM').to_frame().rename(columns={0:'20Y Breakeven'})
    data2 = fred.get_series('T7YIEM').to_frame().rename(columns={0:'7Y Breakeven'})

    df = pd.concat([data, data2], axis=1).reset_index()

    df['index'] = df['index'].astype(str)
    df.set_index('index', inplace = True)
    df.dropna(axis=0, how = 'any', inplace = True)
    # g = sns.lineplot(data=data, x = 'index', y = '20Y Breakeven', color = 'grey')

    # ax2 = plt.twinx()
    # sns.lineplot(data=data2, x = 'index', y = '7Y Breakeven', color="b", ax=ax2)

    # plt.title('Expected Inflation')
    print(df)

    return df



def expected_inflation_10Y():
    fred = fredapi.Fred(api_key=fred_api_key)

    data = fred.get_series('T10YIE').to_frame().rename(columns={0:'10Y Breakeven'})
    data2 = fred.get_series('DGS10').to_frame().rename(columns={0:'10Y Yield'})

    df = pd.concat([data, data2], axis=1).reset_index()

    df['index'] = df['index'].astype(str)
    df.set_index('index', inplace = True)
    df.dropna(axis=0, how = 'any', inplace = True)
    
    print(df)

    return df    