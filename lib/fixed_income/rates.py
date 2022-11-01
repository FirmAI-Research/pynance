import sys, os
import numpy as np
import matplotlib.pyplot as plt
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve, NelsonSiegelCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols, calibrate_nss_ols
from datetime import datetime
import pandas as pd
import numpy as np 
from re import L, X
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import json

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(proj_root)
sys.path.append(proj_root)
from calendar_dates import Calendar
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




















def build_zero_curve():
    ''' The variables NS_ZC and NS_Fwd will give the zero coupon rates and implied forward rates using Nelson Siegel model, similarly the next two lines of code with variables NSS_ZC and NSS_Fwd will give the output using Nelson Siegel Svensson model.
    '''
    #tenors
    t = np.array([0.0,0.5,0.75,1.0,1.5,2.0,3.0,4.0,5.0,6.0,7.0,8.0,10.0])

    #market rates
    y=np.array([0.01,0.02,0.025,0.029,0.03,0.033,0.036,0.039,0.04,0.043,0.045,0.047,0.049])
    curve_fit1, status1 = calibrate_ns_ols(t,y) #NS model calibrate
    curve_fit, status = calibrate_nss_ols(t,y) #NSS model calibrate
    NS_ZC = NelsonSiegelCurve.zero(curve_fit1,t)
    NS_Fwd = NelsonSiegelCurve.forward(curve_fit1,t)

    NSS_ZC = NelsonSiegelSvenssonCurve.zero(curve_fit,t)
    NSS_Fwd = NelsonSiegelSvenssonCurve.forward(curve_fit,t)
    print(NS_ZC)

