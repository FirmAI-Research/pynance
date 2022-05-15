""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
                                            Nasdaq Data Link                                  
  
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

import re
import nasdaqdatalink
from curses import keyname
import json
import os
import pandas as pd
import numpy as np
import requests
import os
import platform
import datetime
from datetime import timezone
from lib.calendar import Calendar
cal = Calendar()
import pytz

utc=pytz.UTC

class Nasdaq:
    """
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31')
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate='2021-12-31', ticker='ZYXI')
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1.csv?api_key=API_KEY?download_type=complete
    mydata = nasdaqdatalink.get("EIA/PET_RWTC_D")

    """    
    iodir =  os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/_tmp/nasdaq_data_link/'  

    # equtiy
    def __init__(self):
        self.authenticate()


    def authenticate(self):        
        fp = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/_tmp/nasdaq_data_link.json'
        with open(fp) as f:
            data = json.load(f)
        #nasdaqdatalink.read_key() 
        nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key'] 
        os.environ["NASDAQ_DATA_LINK_API_KEY"] =  data['nasdaq_api_key']  # NOTE options: nasdaqdatalink.ApiConfig.api_key = data['nasdaq_api_key']   |     nasdaqdatalink.read_key()     
        self.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]


    def calculate_box_plot(self, df, column=None):
        ''' calculat quartiles of an individual array '''
        arr = pd.to_numeric(df[column]).dropna().values
        Q1, median, Q3 = np.percentile(arr, [25, 50, 75])
        IQR = Q3 - Q1

        loval = Q1 - 1.5 * IQR
        hival = Q3 + 1.5 * IQR

        wiskhi = np.compress(arr <= hival, arr)
        wisklo = np.compress(arr >= loval, arr)

        actual_hival = np.max(wiskhi)
        actual_loval = np.min(wisklo)

        outliers_high = np.compress(arr > actual_hival, arr)
        outliers_low = np.compress(arr < actual_loval, arr)
        outliers = [x for x in outliers_high ] + [y for y in outliers_low]

        Qs = [actual_loval, Q1, median, Q3, actual_hival]
        # print(arr)
        # print(Qs)
        return Qs, outliers


    def build_percentiles_frame(self, df):
        datalists = []
        for c in df.columns:
            if c in self.fundamental_cols and c != 'ticker':
                qs, outliers = self.calculate_box_plot(df, column=c)
                datalists.append(qs)
        xdf = pd.DataFrame(datalists, columns=['low', 'Q1', 'median', 'Q3', 'high']).transpose()
        xdf.columns = [x for x in self.fundamental_cols if x != 'ticker']
        xdf.drop(columns=['calendardate'], inplace=True)
        for c in xdf.columns:
            xdf[c] = xdf[c].apply(lambda x : '{:,.2f}'.format(x))
        xdf.reset_index(inplace=True)
        return xdf


    def get_modified_time(self, path_to_file):
        """
        See http://stackoverflow.com/a/39501288/1709587 for explanation.
        """
        import time
        if platform.system() == 'Windows':
            return  time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(os.path.getmtime(path_to_file)))# getmtime= modified, getctime = created
        else:
            stat = os.stat(path_to_file)
            modified = datetime.datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            return modified


class Metrics(Nasdaq):

    name = 'SHARADAR/Daily'

    ticker_cols = ['ticker', 'name','exchange','category','cusips', 'sector', 'industry', 'scalemarketcap', 'scalerevenue', 'currency']

    def __init__(self, ):        
        super().__init__()
        self.authenticate()


    def get(self): # NOTE using prior quarter fundamentals for complete dataset
        df = nasdaqdatalink.get_table(self.name, date=[cal.previous_quarter_end()], paginate=True) ##qopts={"columns":"compnumber"}, date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })
        return df


    def merge_meta_data(self, df):
        tickers = Tickers()
        df_tickers =  tickers.get().dropna(subset=['ticker'])
        df_tickers = df_tickers[self.ticker_cols]
        return df.merge(df_tickers, how='left', on='ticker')

    def metrics_by_sector(self, sector=None):
        metrics = self.get()
        self.sector_df = self.merge_meta_data(metrics)
        self.sector_df.to_excel(f'{self.iodir}/nasdaq_metrics.xlsx')
        self.sector_df =  self.sector_df.loc[self.sector_df.sector == sector]
        self.sector_df.to_excel(f'{self.iodir}/nasdaq_metrics_sector_view.xlsx')
        self.df = self.sector_df
        return self.sector_df




class Fundamentals(Nasdaq):
    """
    5 filter columns:
        ticker, calendardate, lastupdated, dimension, datekey
    full export - https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key=API_KEY

    Notes:
    nasdaqdatalink.get_table('SHARADAR/SF1', calendardate={'gte':'2013-12-31'}, ticker='AAPL')
    qopts={"columns":"compnumber"}, date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })
    """
    name = 'SHARADAR/SF1'

    fundamental_cols = ['ticker','calendardate', 'assets','capex', 'debt', 'depamor', 'ebit', 'ebitda', 'eps','equity', 'ev',  'fcf', 'gp', 'intangibles', 'intexp', 'inventory',  'liabilities', 'ncf', 'netinc', 'opex', 'opinc', 'pb', 'pe',
    'price', 'ps', 'receivables', 'revenue', 'rnd', 'tangibles', 'taxassets', 'taxexp', 'taxliabilities', 'workingcapital',
    'roe','roc','roa']

    ticker_cols = ['name','exchange','category','cusips', 'sector', 'industry', 'scalemarketcap', 'scalerevenue', 'currency']

    def __init__(self, ticker = None, calendardate = None):        
        super().__init__()
        self.authenticate()

        self.calendardate = calendardate


    def get(self): # NOTE using prior quarter fundamentals for complete dataset
        fp = f'{self.iodir}/all_fundamentals.xlsx'

        if not os.path.exists(fp) or (pd.to_datetime(pd.to_datetime(self.get_modified_time(fp)).strftime('%Y-%m-%d')) < pd.to_datetime(pd.to_datetime(utc.localize(cal.today())).strftime('%Y-%m-%d'))): 
            print('File does not exist or has not been updated today. Downloading full query results...')

            print(f'Modified: {pd.to_datetime(self.get_modified_time(fp))}')
            print(f'Today: {cal.today()}')
            df = nasdaqdatalink.get_table(self.name, dimension="MRQ", calendardate=[self.calendardate],  paginate=True) 
            df['shequity'] = df['assets'] - (df['liabilities'] )
            df['roe'] = df['netinc'] / (df['equity'] )
            df['roc'] = df['netinc'] / (df['equity'] + df['debt'])
            df['roa'] = df['netinc'] / df['assets']
            df.to_excel(fp)
        else:
            print('data has been updated today - reading from file')
            df = pd.read_excel(fp)


        return df


    def merge_meta_data(self, df):
        tickers = Tickers()
        df_tickers =  tickers.get().dropna(subset=['ticker'])
        return df.merge(df_tickers, how='left', on='ticker')


    def fundamentals_by_sector(self, sector=None):
        fdmtl = self.get()
        self.sector_df = self.merge_meta_data(fdmtl)
        # self.sector_df.to_excel(f'{self.iodir}/nasdaq_fundamentals.xlsx')
        self.sector_df =  self.sector_df.loc[self.sector_df.sector == sector]
        return self.sector_df


    def view_sector(self):
        self.df = self.sector_df[self.ticker_cols + self.fundamental_cols]
        # self.df.to_excel(f'{self.iodir}/nasdaq_fundamentals_sector_view.xlsx')
        return self.df


    # Misc.
    def curl(self):
        # request_url = f"https://data.nasdaq.com/api/v3/datatables/{self.name}.json?api_key={self.api_key}&dimension=MRQ&ticker=AMZN&filingdate.gte=2022-01-01"
        # res = requests.get(request_url)
        # df  = pd.DataFrame.from_dict(res.json())
        # print(df)
        pass


    # def get_export(self, dimension='MRQ'):
    #     df = pd.read_csv('./vendors/exports/SHARADAR_SF1_Full_Export.csv')
    #     prev_qtr = str(Calendar().previous_quarter_end())
    #     df = df.loc[(df.dimension == dimension) & (df.calendardate == prev_qtr)]
    #     return df





class Tickers(Nasdaq):

    name = 'SHARADAR/TICKERS'

    def __init__(self):
        super().__init__()
        self.authenticate()


    def get(self):

        fp = f'{self.iodir}/all_tickers.xlsx'

        if not os.path.exists(fp) or (pd.to_datetime(pd.to_datetime(self.get_modified_time(fp)).strftime('%Y-%m-%d')) < pd.to_datetime(pd.to_datetime(utc.localize(cal.today())).strftime('%Y-%m-%d')) ): 
            print('Tickers File does not exist or has not been updated today. Downloading full query results...')
            print(f'Modified: {pd.to_datetime(self.get_modified_time(fp))}')
            print(f'Today: {cal.today()}')
            df = nasdaqdatalink.get_table(self.name, table="SF1", paginate=True ) #qopts={"columns":"compnumber"}, date = { 'gte': '2016-01-01', 'lte': '2016-12-31' })
            df = df.loc[(df.isdelisted == 'N') & (df.lastpricedate == max(df.lastpricedate)) ]
            df = df.drop_duplicates(subset=['ticker', 'name'])
            df.to_excel(fp)
        else:
            print('Tickers data has been updated today - reading from file')
            df = pd.read_excel(fp)

        return df


    # def get_export(self, table_name='SF1'):
    #     df = pd.read_csv('./vendors/exports/SHARADAR_TICKERS.csv')
    #     df = df.loc[df.table==table_name]
    #     return df








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





class Insiders(Nasdaq):
    '''
    https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF3.csv?&calendardate=2021-09-30&api_key=-sZL9MEeYDAGRHMhb7Uz&qopts.export=true
        > qopts.export=True bypasses the 10,000 row export limit and provides excel file with link to s3 bucket in cell A1
        > curl link in  A1 to download zip containing all data

    curl https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF2?qopts.export=true&api_key=API_KEY

    '''

    name = 'SHARADAR/SF2'

    def __init__(self):
        super().__init__()
        self.authenticate()

        self.prev_qtr_end = Calendar().previous_quarter_end()



    def curl(self):
        # self.df = nasdaqdatalink.get_table(self.name, paginate=True, filingdate=self.prev_qtr_end ) # FIXME: filingdate grtr equal to 
        # self.df['pctofsharesownedbefore'] = self.df['transactionshares'] / self.df['sharesownedbeforetransaction']
        request_url = f"https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF2.json?api_key={self.api_key}&filingdate.gte=2022-01-01"
        res = requests.get(request_url)
        df  = pd.DataFrame.from_dict(res.json())
        # df = pd.read_json(request_url)
        # print(res)
        # print(type(res))
        # print(res.text)

        # df = pd.read_json(res.text)
        print(df)


        # print(f'[SUCCEESS] Retrieved Insiders data: {len(self.df)} rows')
        # print(self.df.shape)
        # print(self.df.columns)
        # print(self.df.head())
        # return self.df
    


    def direct_by_ticker(self):
        df = self.df.loc[ (self.df.directorindirect=='D') ]
        df = df[['ticker','issuername','ownername','transactionvalue']].groupby(['ticker','issuername','ownername',]).agg({'transactionvalue':'sum'})
        print(df)
        return df


    def ten_percenters(self):
        pass


    
