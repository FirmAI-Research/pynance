from tkinter import N
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from lib.calendar import Calendar

class TreasuryRates:

    def __init__(self):
        pass


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
        return df


    def melt_by_year(self, years = [Calendar().current_year()]):
        df = self.get(years = years)
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)
        return self.melted


    def melt_by_month(self, years = [Calendar().current_year()], n=6):
        df = self.get(years = years)
        df = df.iloc[-n:,:]
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)
        return self.melted


    def plot_curve(self):
        df = self.get()
        df = pd.DataFrame(df.iloc[-1]).transpose()
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)
        # fig, ax = plt.subplots(1, 1)
        # sns.lineplot(x = 'variable', y = 'value', data = self.melted)
        # plt.show()
        return self.melted


    def plot_curve_at_points_in_time(self):
        df = self.get().set_index('date')

        nrows = len(df.index)
        mr_month = df.iloc[-1]
        mr_minus_1_week = df.iloc[nrows-7]      # TODO: not precise; use market day dates
        mr_minus_1_month = df.iloc[nrows-30]

        select_rows = pd.concat([mr_month, mr_minus_1_week, mr_minus_1_month], axis=1).transpose().reset_index().rename(columns = {'index':'date'}).melt(id_vars='date')
        select_rows.value = pd.to_numeric(select_rows.value)
        # fig, ax = plt.subplots(1, 1)
        # sns.lineplot(x = 'variable', y = 'value', hue ='date', data = select_rows)
        # plt.show()
        return select_rows
    
    def change_distribution(self):
        df = self.get().set_index('date')
        weekly_rows = df.iloc[-7:]
        weekly_diff = weekly_rows.diff().dropna(how='all', axis=0).reset_index(drop=False)
        weekly_diff = weekly_diff.melt(id_vars='date')
        return weekly_diff



# tr = TreasuryRates()
# df = tr.get()
# print(df)
# tr.ts_by_years(years =  ['2021', '2022'])
# tr.ts_by_months(years = ['2022'], n=14)
# tr.plot_curve()
# tr.plot_curve_at_points_in_time()
# tr.change_distribution()