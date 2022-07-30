from datetime import datetime
from re import L, X
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import json
from lib.calendar import Calendar

def get(self, years:list= [Calendar().current_year()]):
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
    self.df = df
    return self.df
