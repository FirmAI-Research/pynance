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
        os.environ["NASDAQ_DATA_LINK_API_KEY"] =  data['nasdaq_api_key']  # NOTE options: nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key']   |     nasdaqdatalink.read_key()     


    def get(self, name, ticker=None, date = None):
        # if ticker is None:
        #     return nasdaqdatalink.get_table(name, paginate=True)
        # else:
        #     return nasdaqdatalink.get_table(name, ticker=ticker)
        df = nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31', ticker = 'AMZN', paginate=True)
        df.to_csv('.\\vendors\\exports')




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
        #self.authenticate()


    def get(self):
        df = nasdaqdatalink.get_table(self.name, paginate=True, nrows=10)
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
        df = nasdaqdatalink.get_table(self.name, paginate=True, nrows=10)
        print(df)


    def get_export(self, table_name='SF1'):
        df = pd.read_csv('./vendors/exports/SHARADAR_TICKERS.csv')
        df = df.loc[df.table==table_name]
        return df

