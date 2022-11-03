''' Inflation; Unemployment; GDP; Manufacturing; Home Prices; Implied Expected Inflation; 10 Year breakeven inflation; Fixed Income sectors; Currency; Fwd P/e
 Equity/Bond Market Correlation
'''
import fredapi
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys, os
import json

proj_root = os.path.abspath('')
fp = os.path.join(proj_root, 'secrets.json')
with open(fp) as f:
    data = json.load(f)
fred_api_key = data['fred_api_key'] 

fred = fredapi.Fred(api_key=fred_api_key)


def spy_and_vol():
    data = fred.get_series('VIXCLS').to_frame().rename(columns={0:'Volatlity'}).merge(fred.get_series('SP500').to_frame().rename(columns={0:'S&P500'}), left_index = True, right_index = True)

    data.reset_index(inplace = True)
    g = sns.lineplot(data=data, x = 'index', y = 'S&P500', color = 'orange')

    ax2 = plt.twinx()
    sns.lineplot(data=data, x = 'index', y = 'Volatlity', color="b", ax=ax2)

    plt.title('S&P and Volitlity')

    None


def inflation():
    inflation = fred.get_series('PCE').to_frame().rename(columns={0:'PCE'}).merge( 
    fred.get_series('PCEPILFE').to_frame().rename(columns={0:'PCEPILFE'}), left_index=True, right_index=True).merge( # Core PCE 
    fred.get_series('CPIAUCSL').to_frame().rename(columns={0:'CPIAUCSL'}), left_index=True, right_index=True).merge( # CPI 
    fred.get_series('CPILFESL').to_frame().rename(columns={0:'CPILFESL'}), left_index=True, right_index=True) # Core CPI
    inflation_yoy = (inflation / inflation.shift(12)) -1
    inflation_yoy.iloc[-36:].plot(title='YoY Inflation')


def consumer_sentiment():
    data = fred.get_series('UMCSENT').to_frame().iloc[-200:].rename(columns={0:'UMich Consumer Sentiment'})
    data.plot(title = 'UMich Consumer Sentiment')


def fomc_projections():
    data = fred.get_series('FEDTARMD').to_frame().rename(columns={0:'Median'}).merge( 
    fred.get_series('FEDTARRL').to_frame().rename(columns={0:'Low'}), left_index=True, right_index=True).merge( 
    fred.get_series('FEDTARRH').to_frame().rename(columns={0:'High'}), left_index=True, right_index=True).dropna(how='all')
    data.plot(title = 'FOMC Fed Funds Projections')

def industrial_prod():
    data = fred.get_series('INDPRO').to_frame().iloc[-100:].rename(columns={0:'Industrial Production'})
    data.plot(title = 'Industrial Production')


def corporate_profits():
    data = fred.get_series('CP').to_frame().iloc[-50:].rename(columns={0:'Corporate profits'})
    data.plot(title = 'Corporate Profits')

def gdp():
    data = fred.get_series('GDP').to_frame().rename(columns={0:'GDP'})
    data.plot(title = 'GDP')