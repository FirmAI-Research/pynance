""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                                            Nasdaq Data Link                                  
  
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

import nasdaqdatalink
from curses import keyname
import json
import os
import pandas as pd
import numpy as np

from controller.calendar import Calendar

class Nasdaq:
    """
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31')
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31', ticker='ZYXI')
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.csv?api_key=API_KEY?download_type=complete
    mydata = nasdaqdatalink.get("EIA/PET_RWTC_D")

    """    

    # equtiy
    def __init__(self):
        self.authenticate()


    def authenticate(self):        
        with open('../secrets.json') as f:
            data = json.load(f)
        #nasdaqdatalink.read_key() 
        nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key'] 
        os.environ["NASDAQ_DATA_LINK_API_KEY"] =  data['nasdaq_api_key']  # NOTE options: nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key']   |     nasdaqdatalink.read_key()     






class CoreUsFundamentals(Nasdaq):
    """
    view:
        AR - As reported
        MR - Most reecent reported

    time dimensions:
        Y - Annual
        Q - Quarter
        T - Trailing Twelve Months

    filters: 
        1 - ticker
        2 - calendardate
        3 - lastupdated
        4 - dimension
        5 - datekey


    full export - https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key=API_KEY

    """

    name = 'SHARADAR/SF1'

    def __init__(self):
        super().__init__()
        self.authenticate()


    def get(self):
        df = nasdaqdatalink.get_table(self.name, paginate=True, dimension="MRY", ticker='AMZN')
        print(df)


    def get_export(self, dimension='MRQ'):
        df = pd.read_csv('./vendors/exports/SHARADAR_SF1_Full_Export.csv')
        prev_qtr = str(Calendar().previous_quarter_end())
        df = df.loc[(df.dimension == dimension) & (df.calendardate == prev_qtr)]
        return df

    def merge_meta_data(self, df_core):
        tickers = Tickers()
        df_tickers =  tickers.get_export().dropna(subset=['ticker'])
        df = df_core.merge(df_tickers, how='left', on='ticker')
        return df

    def parse_seector_industry(self, sector:str=None, industry:str=None):
        pass



class Tickers(Nasdaq):

    name = 'SHARADAR/TICKERS'

    def __init__(self):
        super().__init__()
        #self.authenticate()


    def get(self):
        df = nasdaqdatalink.get_table(self.name, paginate=True)
        print(df)


    def get_export(self, table_name='SF1'):
        df = pd.read_csv('./vendors/exports/SHARADAR_TICKERS.csv')
        df = df.loc[df.table==table_name]
        return df


class CoreUSInstitutionalInvestors(Nasdaq):
    '''
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF3.csv?&calendardate=2021-09-30&api_key=-sZL9MEeYDAGRHMhb7Uz&qopts.export=true
        > qopts.export=True bypasses the 10,000 row export limit and provides excel file with link to s3 bucket in cell A1
        > curl link in  A1 to download zip containing all data
    '''

    name = 'SHARADAR/SF3'

    def __init__(self):
        super().__init__()
        self.authenticate()


    def get(self, date, institution=None):
        # self.df = nasdaqdatalink.bulkdownload("SF3")
        df = nasdaqdatalink.get_table(self.name, paginate=True, calendardate=date, investorname=institution)
        return df
        

    def get_export(self, fp=None):
        if fp is None:
            fp = './vendors/exports/SHARADAR_SF3_Full_Export.csv'
        
        self.df = pd.read_csv(fp)
        return self.df


    def list_all_institutions(self):
        if not hasattr(self, 'df'):
            self.get_export()
        return sorted(self.df.investorname.unique().tolist(), reverse=False)


    def group_by_ticker(self):
        df = self.df[['ticker', 'units', 'price']].groupby('ticker').agg({"units":np.sum, "price":np.mean})
        df['value'] = (df['units'] * df['price']) / 1000000000
        df = df.sort_values(by='value', ascending=False)
        df.to_csv('./instituational_investors_qtr_end.csv')
        return df
    

    def group_by_institution(self):
        df = self.df[['investorname', 'ticker', 'units', 'price']].groupby(['investorname', 'ticker']).agg({"units":np.sum, "price":np.mean})
        df['value'] = (df['units'] * df['price']) / 1000000
        return df


    def qtr_over_qtr_change(self, qtr_start, qtr_end):
        # quarter over quarter
        dates =  Calendar().quarter_end_list('2020-12-31', '2021-12-31')

        frames = []
        for date in dates:
            df = self.get(date = date, institution='BLACKROCK INC')
            df = df.sort_values(by=['value'], ascending=False)
            df = df.iloc[:11]
            frames.append(df)
            print(df.head())
            print(df.shape)
        df = pd.concat(frames)
        # sns.lineplot(data=df, x="calendardate", y="value", hue='ticker')
        # plt.show()



class Insiders:


    def __init__():
        pass