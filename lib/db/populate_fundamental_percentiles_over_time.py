import pandas as pd
import numpy as np
import nasdaqdatalink
import sys, os
cwd = os.getcwd()
sys.path.append(os.path.dirname(cwd))
from lib.nasdaq import Fundamentals, Metrics, Tickers, Nasdaq
from lib.calendar import Calendar
cal = Calendar()
from dateutil.relativedelta import relativedelta
from postgres import Postgres

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


import populate_fundamentals as popfun

'''
calculate quartiles for every metric across each sector and industry for all historical time periods
'''

def init():

    calendardates = cal.quarter_end_list('2018-12-31', cal.previous_quarter_end())

    tickers = Tickers().get() 
    tickers = tickers.loc[ ( pd.to_datetime(tickers.lastpricedate) >= pd.to_datetime(cal.previous_quarter_end()) ) & (tickers.currency == 'USD')]
    tickers = tickers[['ticker','name','cusips','sector','industry', 'sicsector', 'sicindustry',  'famaindustry']]


    lower_frames, median_frames, upper_frames = [], [], []
    for date in calendardates:
        fun = Fundamentals(calendardate=date)
        fundamentals = fun.get()
        df = tickers.merge(fundamentals, how='left', on='ticker')

        unique_sectors_list = df.sector.unique().tolist()

        for sector_str in unique_sectors_list:

            try:
                asector = df.loc[df.sector == sector_str]
                sector_prcentiles = popfun.build_percentiles_frame(asector)

                lower_quart = sector_prcentiles.loc[sector_prcentiles['index'] == 'Q1'].drop('index', axis=1)
                lower_quart['sector'] = sector_str
                lower_quart['date'] =date

                median = sector_prcentiles.loc[sector_prcentiles['index'] == 'median'].drop('index', axis=1)
                median['sector'] =sector_str
                median['date'] = date
                
                upper_quart = sector_prcentiles.loc[sector_prcentiles['index'] == 'Q3'].drop('index', axis=1)
                upper_quart['sector'] =sector_str
                upper_quart['date'] =date
                
                lower_frames.append(lower_quart)
                median_frames.append(median)
                upper_frames.append(upper_quart)

                print('...Calculated values for {} '.format(sector_str))
            
            except Exception as e:
                print(f'*****[ERROR] @ {sector_str} ***** {e}')            

        lower_df = pd.concat(lower_frames)
        median_df = pd.concat(median_frames)
        upper_df = pd.concat(upper_frames)

        pg = Postgres()
        engine = pg.engine

        lower_df.to_sql('Quartile_Values_Over_Time_Lower_by_Sector', engine, if_exists='replace')
        median_df.to_sql('Quartile_Values_Over_Time_Median_by_Sector', engine, if_exists='replace')
        upper_df.to_sql('Quartile_Values_Over_Time_Upper_by_Sector', engine, if_exists='replace')

        print('[SUCCESS] Populated Quartile values for {}'.format( date))



    lower_frames, median_frames, upper_frames = [], [], []
    for date in calendardates:
        fun = Fundamentals(calendardate=date)
        fundamentals = fun.get()
        df = tickers.merge(fundamentals, how='left', on='ticker')

        unique_industries_list = df.industry.unique().tolist()
        for industry_str in unique_industries_list:

            try:
                asector = df.loc[df.industry == industry_str]
                sector_prcentiles = popfun.build_percentiles_frame(asector)

                lower_quart = sector_prcentiles.loc[sector_prcentiles['index'] == 'Q1'].drop('index', axis=1)
                lower_quart['industry'] = industry_str
                lower_quart['date'] =date

                median = sector_prcentiles.loc[sector_prcentiles['index'] == 'median'].drop('index', axis=1)
                median['industry'] =industry_str
                median['date'] = date
                
                upper_quart = sector_prcentiles.loc[sector_prcentiles['index'] == 'Q3'].drop('index', axis=1)
                upper_quart['industry'] =industry_str
                upper_quart['date'] =date
                
                lower_frames.append(lower_quart)
                median_frames.append(median)
                upper_frames.append(upper_quart)

                print('...Calculated values for {} '.format(industry_str))
            
            except Exception as e:
                print(f'*****[ERROR] @ {industry_str} ***** {e}')            

        lower_df = pd.concat(lower_frames)
        median_df = pd.concat(median_frames)
        upper_df = pd.concat(upper_frames)

        pg = Postgres()
        engine = pg.engine

        lower_df.to_sql('Quartile_Values_Over_Time_Lower_by_Industry', engine, if_exists='replace')
        median_df.to_sql('Quartile_Values_Over_Time_Median_by_Industry', engine, if_exists='replace')
        upper_df.to_sql('Quartile_Values_Over_Time_Upper_by_Industry', engine, if_exists='replace')

        print('[SUCCESS] Populated Quartile values for {}'.format( date))


init()