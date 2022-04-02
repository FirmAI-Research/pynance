from tkinter import N
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


class TreasuryRates:

    def __init__(self):
        pass


    def get(self, years:list= ['2022']):
        ''' parse US Treasury.gov website for yield curve rates returning xml content for a list of years and concatenating
        https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2022
        '''

        columns =  ['1Month','2Month','3Month', '6Month','1Year','2Year','3Year','5Year','7Year','10Year','20Year','30Year']
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
        
        return df


    def _lineplot(self, data):
        sns.lineplot(x = 'date', y = 'value', hue = 'variable', data = data)
        # plt.ylim(0, 500)
        plt.show()


    def ts_years(self, years = ['2022']):
        df = self.get(years = years)
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)
        self._lineplot(data = self.melted)


    def ts_months(self, years = ['2022'], n=6):
        df = self.get(years = years)
        df = df.iloc[-n:,:]
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)
        self._lineplot(data = self.melted)


    def plot_curve(self):
        df = self.get()
        df = pd.DataFrame(df.iloc[-1]).transpose()
        self.melted = df.melt(id_vars='date')
        self.melted.value = pd.to_numeric(self.melted.value)

        print(self.melted)

        fig, ax = plt.subplots(1, 1)
        sns.lineplot(x = 'variable', y = 'value', data = self.melted)

        plt.show()


    def plot_curve_multiple_years(self):
        pass

    


tr = TreasuryRates()
# tr.ts_years(years =  ['2021', '2022'])
# tr.ts_months(years = ['2022'], n=6)
tr.plot_curve()