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
        '''
        TODO; pass multiple dates and specific institution - return df
        '''
        df = nasdaqdatalink.get_table(self.name, paginate=True, calendardate=date, investorname=institution)
        print(f'[SUCCEESS] Retrieved {institution} data for {date}: {len(df)} rows')
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


    def favorite_institutions(self):
        return [ 'BRIDGEWATER ASSOCIATES LP', 'BLACKROCK INC','VANGUARD GROUP INC','JANUS HENDERSON GROUP PLC','CITADEL ADVISORS LLC',
            'UBS ASSET MANAGEMENT AMERICAS INC','GOLDMAN SACHS GROUP INC','MORGAN STANLEY','BARCLAYS PLC',
            'SUSQUEHANNA INTERNATIONAL GROUP LLP', 'TWO SIGMA INVESTMENTS LP','ELLIOTT INVESTMENT MANAGEMENT LP',
            'BROOKFIELD ASSET MANAGEMENT INC','BERKSHIRE ASSET MANAGEMENT LLC','BERKSHIRE HATHAWAY INC',
            'OAKTREE CAPITAL MANAGEMENT LP' ]


    def group_by_ticker(self):
        df =  self.df
        df['count'] = 0
        df = df[['ticker', 'units', 'price', 'count']].groupby('ticker').agg({"units":np.sum, "price":np.mean, "count":np.size})
        df['value'] = (df['units'] * df['price']) / 1000000000
        df = df.sort_values(by='value', ascending=False)
        df =  df[['price', 'units', 'value', 'count']]  #  ticker is index
        return df
    

    def group_by_institution(self):
        df = self.df[['investorname', 'ticker', 'units', 'price']].groupby(['investorname', 'ticker']).agg({"units":np.sum, "price":np.mean})
        df['value'] = (df['units'] * df['price']) / 1000000
        return df


    def time_series_range(self, institution, qtr_start, qtr_end):
        self.institution = institution
        dates =  Calendar().quarter_end_list(qtr_start,  qtr_end)
        frames = []
        for date in dates:
            df = self.get(date = date, institution=institution) # FIXME; terrible way to call. get all data in one call. cmon...
            df = df.sort_values(by=['value'], ascending=False)
            frames.append(df)
        self.ts = pd.concat(frames).reset_index(drop=True)
        self.ts.value = self.ts.value.astype(float) / 1000000
        self.ts['_ix'] = self.ts.ticker.astype(str) + '_' + self.ts.securitytype.astype(str)
        return self.ts


    def change(self):
        ''' must call time_series_range first to createe self.ts attribute; NOTE that it includes api calls through get()
        '''
        dates = sorted(self.ts.calendardate.unique().tolist()) # last value is most recent date;  sorting converts  datee to int
        dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in dates]  # after sorting, convert back to date
        print(dates)

        dfx = self.ts.loc[self.ts.calendardate == dates[-1]].set_index('_ix') # most recent date's data
        dfy = self.ts.loc[self.ts.calendardate == dates[0]].set_index('_ix')  # earliest dates data

        dfz = pd.DataFrame()
        dfz['ticker'] = dfx['ticker']
        dfz['securitytype'] = dfx['securitytype']
        dfz[dates[-1]] = dfx['value']
        dfz[dates[0]] = dfy['value']
        dfz['pct'] = (dfx['value'] - dfy['value']) / dfy['value']


        df_increase = dfz.sort_values(by = ['pct'], ascending=False)
        df_decrease = dfz.sort_values(by = ['pct'], ascending=True)
        return (df_increase, df_decrease)





class Insiders:


    def __init__():
        pass