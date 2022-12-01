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

def get_credit_spreads():
    # Moodys seasoned bond yields
    fred = fredapi.Fred(api_key=fred_api_key)
    credit_spreads = fred.get_series('AAA').to_frame().rename(columns={0:'AAA'}).merge( 
        fred.get_series('BAA').to_frame().rename(columns={0:'BAA'}), left_index=True, right_index=True).merge(
        fred.get_series('BAMLH0A0HYM2EY').to_frame().rename(columns={0:'HY'}), left_index=True, right_index=True)
    credit_spreads.reset_index(inplace = True)
    credit_spreads['index'] = credit_spreads['index'].astype(str)
    credit_spreads = credit_spreads.dropna(axis=0, how = 'any').set_index('index').iloc[-48:]
        # melt = credit_spreads.reset_index().melt(id_vars=['index'])
    # plt.figure(figsize = (20,5))

    # g = sns.lineplot(data=melt, x = 'index', y = 'value', hue = 'variable')
    # plt.xticks(rotation=45)
    # None
    return credit_spreads
