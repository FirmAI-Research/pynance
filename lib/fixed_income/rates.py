import sys, os
import numpy as np
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
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
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)
from calendar_dates import Calendar
import fredapi
import sys, os, json
import seaborn as sns


proj_root = os.path.abspath('')
fp = os.path.join(proj_root, 'secrets.json')
with open(fp) as f:
    data = json.load(f)
fred_api_key = data['fred_api_key'] 


import numeric
cal = Calendar()

class Treasuries:

    def __init__(self, years:list= [Calendar().current_year()]):
        
        self.get(years)


    def get(self, years):
        ''' parse US Treasury.gov website for yield curve rates returning concatenated xml content for a list of years
        https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2022
        '''

        columns =  ['1 Month','2 Month','3 Month', '6 Month','1 Year','2 Year','3 Year','5 Year','7 Year','10 Year','20 Year','30 Year']
        identifiers = ['d:NEW_DATE', 'd:BC_1MONTH', 'd:BC_2MONTH', 'd:BC_3MONTH', 'd:BC_6MONTH', 'd:BC_1YEAR', 'd:BC_2YEAR', 'd:BC_3YEAR', 'd:BC_5YEAR', 'd:BC_7YEAR', 'd:BC_10YEAR', 'd:BC_20YEAR', 'd:BC_30YEAR']
        
        frames = []
        for year in years:

            url = f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}'
            document = requests.get(url)
            soup= BeautifulSoup(document.content,"lxml-xml")
            frames_year = []

            for content in soup.find_all('m:properties'):

                vlist = []
                for text_value in identifiers:
                    vlist.append(content.find(text_value).text)

                df = pd.DataFrame(columns = columns).transpose()
                df[vlist[0]] = vlist[1:]
                frames_year.append(df.transpose())

            df = pd.concat(frames_year)
            frames.append(df)

        df = pd.concat(frames).reset_index(drop=False).rename(columns = {'index':'date'})  
        df.date = pd.to_datetime(df.date)
        for c in df.columns:
            if c != 'date':
                df[c] = pd.to_numeric(df[c])       
        
        df.date = [d.strftime('%Y-%m-%d') for d in df.date]

        df.set_index('date', inplace = True)

        self.df = df
        
        return self.df




    def build_zero_curve(self):
        ''' The variables NS_ZC and NS_Fwd will give the zero coupon rates and implied forward rates using Nelson Siegel model, similarly the next two lines of code with variables NSS_ZC and NSS_Fwd will give the output using Nelson Siegel Svensson model.
        '''
        #tenors
        # t = np.array([0.0,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0])
        t = np.array([0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0])
        
        #market rates
        # y=np.array([0.01,0.02,0.025,0.029,0.03,0.033,0.036,0.039,0.04,0.043,0.045,0.047,0.049])
        y = self.df.iloc[-1].to_numpy()

        curve_fit1, status1 = calibrate_ns_ols(t,y) #NS model calibrate
        curve_fit, status = calibrate_nss_ols(t,y) #NSS model calibrate
        NS_ZC = NelsonSiegelCurve.zero(curve_fit1,t)
        NS_Fwd = NelsonSiegelCurve.forward(curve_fit1,t)

        NSS_ZC = NelsonSiegelSvenssonCurve.zero(curve_fit,t)
        NSS_Fwd = NelsonSiegelSvenssonCurve.forward(curve_fit,t)
        print(NS_ZC)
        return NSS_ZC, NSS_Fwd


    def rates_map(self):
        import seaborn as sns
        cmap = sns.diverging_palette(20, 230, as_cmap=True)
        sns.heatmap(self.df.iloc[-10:], cmap = cmap, annot=True)


    def rates_change_map(self):
        cmap = sns.diverging_palette(20, 230, as_cmap=True)
        sns.heatmap(self.df.iloc[-10:].diff(axis=0), cmap = cmap, annot=True)

    def change_since(self):
        import pandas as pd
        delta = pd.DataFrame()
        delta['5 days'] = self.df.iloc[-1] - self.df.iloc[-5]
        delta['10 days'] = self.df.iloc[-1] - self.df.iloc[-10]
        delta['20 days'] = self.df.iloc[-1] - self.df.iloc[-20]
        delta['60 days'] = self.df.iloc[-1] - self.df.iloc[-60]
        delta = delta.T[::-1]
        delta.index.name = 'Change Since'

        melt = delta.reset_index().melt(id_vars = ['Change Since'])
        sns.barplot(data = melt, hue = 'Change Since', x = 'variable', y='value')
        
        return delta


    def market_correlations(self, join, title, key="SPY"):
        import numpy as np
        fig, axes = plt.subplots(1, 4)
        sns.set_style('whitegrid')

        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # 20 day corr
        corr = join.iloc[-20:].corr()[key].to_frame().drop(key, axis=0)
        g = sns.heatmap(corr, ax = axes[0],  cmap=cmap, square=True, linewidths=.5, annot = True, cbar = True,  cbar_kws={"shrink": .5})

        g.set_title('20 day Correlation')

        g.set_yticklabels(g.get_yticklabels(), rotation=45)


        # 60 day corr

        corr = join.iloc[-60:].corr()[key].to_frame().drop(key, axis=0)
        g = sns.heatmap(corr,  ax = axes[1], cmap=cmap, square=True, linewidths=.5, annot = True, cbar = True, cbar_kws={"shrink": .5} )

        g.set_title('60 day Correlation')

        g.set_yticklabels(g.get_yticklabels(), rotation=45)

        fig.suptitle('Correlation of US Treasuries and S&P 500')


        # 252 day corr
        corr = join.iloc[-252:].corr()[key].to_frame().drop(key, axis=0)
        g = sns.heatmap(corr,  ax = axes[2], cmap=cmap, square=True, linewidths=.5, annot = True, cbar = True, cbar_kws={"shrink": .5} )

            
        g.set_title('252 day Correlation')

        g.set_yticklabels(g.get_yticklabels(), rotation=45)
        
        # 5 Year (1260 d) corr
        corr = join.iloc[-1260:].corr()[key].to_frame().drop(key, axis=0)
        g = sns.heatmap(corr,  ax = axes[3], cmap=cmap, square=True, linewidths=.5, annot = True, cbar = True, cbar_kws={"shrink": .5} )

        g.set_title('5 Year (1260 day) Correlation')

        g.set_yticklabels(g.get_yticklabels(), rotation=45)
        
        fig.suptitle(title, fontsize=16)

        fig.tight_layout()
        sns.despine()
        

    def stock_bond_correlation(self):
        data = yf.download(f"TLT AGG BND SPY", start="2022-01-01", end=cal.today())['Adj Close']
        roll = data['SPY'].rolling(10).corr(data[['BND','AGG','TLT']])
        roll.reset_index(inplace=True)
        roll.Date = [d.strftime('%Y-%m-%d') for d in roll.Date]
        data.reset_index(inplace = True)
        data.Date = [d.strftime('%Y-%m-%d') for d in data.Date]

        plt.rcParams["figure.figsize"] = (20,6)

        g = sns.barplot(data=roll, x = 'Date', y = 'BND', color = 'grey')

        ax2 = plt.twinx()
        sns.lineplot(data=data, x = 'Date', y = 'SPY', color="b", ax=ax2)

        plt.title('Rolling 10 day correlation of SPY and BND')

        g.set_xticklabels(g.get_xticklabels(), rotation=45)
        for label in g.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        plt.locator_params(nbins=50)


    def stock_treasury_correlation(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from calendar_dates import Calendar
        import yfinance as yf
        plt.rcParams["figure.figsize"] = (20,6)

        cal = Calendar()
        spy = yf.download(f"SPY", start="2017-01-01", end=cal.today())['Adj Close'].to_frame().rename(columns = {'Adj Close':'SPY'}).reset_index().rename(columns = {'Date':'date'})
        spy.date = [d.strftime('%Y-%m-%d') for d in spy.date]
        spy.set_index('date', inplace = True)
        join = self.df.merge(spy, how = 'inner', left_index = True, right_index = True)

        self.market_correlations(join, title = 'Stock/Treasury Correlation')


    def tens_twos_spread(self):
        import pandas as pd
        # overlay recession periods

        raw = self.df.reset_index()
        raw.date = pd.to_datetime(raw.date)
        spread = raw[['date','10 Year', '2 Year']]
        spread['Spread'] = spread['10 Year'] - spread['2 Year']
        g = sns.lineplot(data=spread, x = 'date', y = 'Spread')
        plt.xticks(rotation=45)
        plt.axhline(0, c='black')
        plt.title('10s/2s Spread')

        for label in g.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

    
    def breakeven_inflation(self):
        import sys, os, json
        import fredapi
        import matplotlib.pyplot as plt
        import seaborn as sns
        proj_root = os.path.abspath('')
        fp = os.path.join(proj_root, 'secrets.json')
        with open(fp) as f:
            data = json.load(f)
        fred_api_key = data['fred_api_key'] 

        # Moodys seasoned bond yields
        fred = fredapi.Fred(api_key=fred_api_key)
        data = fred.get_series('T10YIE').to_frame().rename(columns={0:'10Y Breakeven'})
        data.plot(title='10 Year Breakeven Inflation Rate')

    

    def expected_inflation(self):

        # Moodys seasoned bond yields
        fred = fredapi.Fred(api_key=fred_api_key)
        data = fred.get_series('T20YIEM').to_frame().rename(columns={0:'20Y Breakeven'}).reset_index()
        data2 = fred.get_series('T7YIEM').to_frame().rename(columns={0:'7Y Breakeven'}).reset_index()

        g = sns.lineplot(data=data, x = 'index', y = '20Y Breakeven', color = 'grey')

        ax2 = plt.twinx()
        sns.lineplot(data=data2, x = 'index', y = '7Y Breakeven', color="b", ax=ax2)

        plt.title('Expected Inflation')

        
    def credit_spreads(self):
        # Moodys seasoned bond yields
        fred = fredapi.Fred(api_key=fred_api_key)
        credit_spreads = fred.get_series('AAA').to_frame().rename(columns={0:'AAA'}).merge( 
            fred.get_series('BAA').to_frame().rename(columns={0:'BAA'}), left_index=True, right_index=True).merge(
            fred.get_series('BAMLH0A0HYM2EY').to_frame().rename(columns={0:'HY'}), left_index=True, right_index=True)
        credit_spreads.tail()

        melt = credit_spreads.reset_index().melt(id_vars=['index'])
        plt.figure(figsize = (20,5))

        g = sns.lineplot(data=melt, x = 'index', y = 'value', hue = 'variable')
        plt.xticks(rotation=45)
        None