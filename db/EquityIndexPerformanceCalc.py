#!/usr/bin/env python
# coding: utf-8

# Market Cap weighted performance of Equity Sectors (5-,10-,30-,90-, horizontal bar chart)
# 5-,10-,30-,90- losers joined with Fundamentals Percentile Ranks;
# 

# In[1]:


from enum import Enum
import sys, os
import nasdaqdatalink
import pandas as pd 
import numpy as np
from scipy.stats.mstats import gmean
from sqlalchemy import create_engine
from pathlib import Path

proj_root = Path(__file__).resolve().parent.parent
sys.path.append(os.path.join(proj_root, 'lib'))
sys.path.append(os.path.join(proj_root, 'lib', 'equity'))

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar
from numeric import custom_formatting

from calendar_dates import Calendar
from fundamentals import Fundamentals, Columns
cal = Calendar()

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.4f}'.format


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
cnxn = engine.connect()




# ## join data

# In[2]:


def get_ticker_data(date):
    # Get prices by ticker for index start date with shares outstanding and sector/industry
    ndq = nasdaq.Nasdaq()

    df_fun = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", calendardate = cal.previous_quarter_end(pd.to_datetime(date)).strftime('%Y-%m-%d'),  paginate=True) # All MRQ periods; One Ticker

    prev_bd = cal.previous_market_day(cal.today()).strftime('%Y-%m-%d')

    prices = nasdaqdatalink.get_table(Sharadar.PRICES.value, date = date, paginate=True) 

    tick = nasdaq.Tickers()
    df_prof = tick.full_export(curl = False) #

    df = prices.set_index('ticker') \
        .merge(df_prof.set_index('ticker'), how='inner', left_index=True, right_index=True) \
        .merge(df_fun.set_index('ticker'), how='inner', left_index=True, right_index=True).reset_index()

    df = df[['ticker','date','open','closeadj','sector','industry','sharesbas']]

    return df 


# # Construct Initial Index (Determine Index Divisor)
def construct_initial_index(df, sector):
    # Market Cap Weighted Index - This method calculates the index weightings of each constituent and the index divisor used for subsequent days.
    # https://www.ftserussell.com/education-center/calculating-index-values
    
    baseValue = 100

    totalShares = 1000
    
    df['mktCap'] = df.closeadj * df.sharesbas               # the market cap of the company on the date the index was initialy constructed
    
    df['indexCap'] = df.mktCap.sum()                        # the total market cap of the index
    
    df['ratio'] = df.mktCap / df.indexCap                   # the percent of index market cap attributable to a security
    
    df['indexWeight'] = df.ratio * 100                      # weighting of an individual security in the index
    
    df['shares'] = df.indexWeight * totalShares             # number of shares for an individual security
    
    df['indexMarketValue'] = df.closeadj * df.shares        # market value of an individual security with in the index
    
    indexDivisor = df.indexMarketValue.sum() / baseValue   
    
    return indexDivisor, df



def initial_index_calculation(date = '01-01-2022'):
    sector_frames = []
    indexDivisors = {}
    ticker_data = get_ticker_data(date = cal.previous_quarter_end(pd.to_datetime(date)))
    sectors = ticker_data.sector.unique().tolist()

    for sector in sectors:

        print(sector)

        constituents = ticker_data[ticker_data.sector == sector] 

        indexDivisor, sectorDf = construct_initial_index(constituents, sector)

        indexDivisors[sector] = indexDivisor  

        dataForSql = sectorDf[['ticker','date','sector', 'mktCap','indexCap', 'ratio','indexWeight','shares','indexMarketValue']]
        sector_frames.append(dataForSql)

    sectorIndexData = pd.concat(sector_frames, axis=0)
    sectorIndexData.to_sql(con=cnxn, if_exists='replace', name = 'EqSectorIx', index = False)

    divisorDf = pd.DataFrame.from_dict(indexDivisors, orient = 'index').reset_index()
    divisorDf.columns = ['Sector', 'IndexDivisor']
    divisorDf.to_sql(con=cnxn, if_exists='replace', name = 'EqSectorIxDivisors', index = False)

    print(sectorIndexData)
    print(divisorDf)



def calculate_index_by_date(date):

    def calculate(date):

        ticker_data = get_ticker_data(date = cal.previous_quarter_end(pd.to_datetime(date)))

        sectors = ticker_data.sector.unique().tolist()

        data = {}
        for sector in sectors:
            
            constituents = ticker_data[ticker_data.sector == sector] 
            
            indexDivisor, sectorDf = construct_initial_index(constituents, sector)
            
            constituents.set_index('ticker').merge(sectorDf[['ticker','shares']].set_index('ticker'))
            
            prices = nasdaqdatalink.get_table(Sharadar.PRICES.value, date = date, paginate=True)
                
            prices = prices[prices.ticker.isin(constituents.ticker.tolist())]
            
            prices = prices.set_index('ticker').merge(constituents.set_index('ticker')[['shares']], left_index = True, right_index=True).reset_index()
            
            prices['indexMarketValue'] = prices.closeadj * prices.shares
            
            ixDivisors = pd.read_sql(f"select * from EqSectorIxDivisors where Sector = '{sector}' ", cnxn)

            indexValue  = prices.indexMarketValue.sum() / ixDivisors['IndexDivisor'].iloc[0]

            data[sector] = [indexValue]

        print(data)

        return data


    if isinstance(date, list):
        frames = []
        for d in date:
            print(d)
            res = calculate(d)
            df = pd.DataFrame(res)
            frames.append(df)
        df = pd.concat(frames, axis=0)
        df.index = date    
    else:
        res = calculate(date)
        df = pd.DataFrame(res)
        df.index = [date]

    return df



""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Initial                                                                                                          │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
#  Use this method to recalculate the IndexDivisors table 

# initial_index_calculation(date = '01-01-2022')

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Run Date Range to append  index values for new dates                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# dates = [x.strftime('%Y-%m-%d') for x in pd.date_range(start='01/01/2022', end='12/02/2022')] # use pandas market calendar
df = calculate_index_by_date(date = '12-03-2022') 
# df = calculate_index_by_date(date = dates) # retrieves most recent quarter end date for fundamental data from date provided; uses actual date to retreive prices
# df.reset_index().to_sql(con=cnxn, if_exists='replace', name = 'EqSectorIxPerf', index = False)
df.reset_index().to_sql(con=cnxn, if_exists='append', name = 'EqSectorIxPerf', index = False)
print(df.reset_index())