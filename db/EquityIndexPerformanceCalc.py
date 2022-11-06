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

proj_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(''))), 'pynance')
print(proj_root)
sys.path.append(proj_root)

import nasdaq_data_link as nasdaq
from nasdaq_data_link import Sharadar
from numeric import custom_formatting

from calendar_dates import Calendar
from lib.equity.fundamentals import Fundamentals, Columns
cal = Calendar()

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.4f}'.format


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
cnxn = engine.connect()


RECALC = False
indexDivisors = {}


# ## join data

# In[2]:


# Get prices by ticker for index start date with shares outstanding and sector/industry
ndq = nasdaq.Nasdaq()

df_fun = nasdaqdatalink.get_table(Sharadar.FUNDAMENTALS.value, dimension="MRQ", calendardate = cal.prior_quarter_end().strftime('%Y-%m-%d'),  paginate=True) # All MRQ periods; One Ticker

prev_bd = cal.previous_market_day(cal.today()).strftime('%Y-%m-%d')

prices = nasdaqdatalink.get_table(Sharadar.PRICES.value, date = '2022-01-03', paginate=True) 


tick = nasdaq.Tickers()
df_prof = tick.full_export(curl = False) # Set curl = True if data should be refreshed

df = prices.set_index('ticker') \
    .merge(df_prof.set_index('ticker'), how='inner', left_index=True, right_index=True) \
    .merge(df_fun.set_index('ticker'), how='inner', left_index=True, right_index=True).reset_index()

df = df[['ticker','date','open','closeadj','sector','industry','sharesbas']]


# # Construct Initial Index (Determine Index Divisor)

# In[3]:


def construct_initial_index(df, sector):
    # Market Cap Weighted Index
    # https://www.ftserussell.com/education-center/calculating-index-values
    
    baseValue = 100
    totalShares = 1000
    
    df['mktCap'] = df.closeadj * df.sharesbas
    
    df['indexCap'] = df.mktCap.sum()
    
    df['ratio'] = df.mktCap / df.indexCap
    
    df['indexWeight'] = df.ratio * 100
    
    df['shares'] = df.indexWeight * totalShares
    
    df['indexMarketValue'] = df.closeadj * df.shares
    
    indexDivisor = df.indexMarketValue.sum() / baseValue   

    
    return indexDivisor, df


# In[4]:


# Skip if recalc not needed
sector_frames = []
if RECALC:

    sectors = df.sector.unique().tolist()

    for sector in sectors:

        print(sector)

        constituents = df[df.sector == sector] 

        indexDivisor, sectorDf = construct_initial_index(constituents, sector)

        
        indexDivisors[sector] = indexDivisor  

        
        dataForSql = sectorDf[['ticker','date','sector', 'mktCap','indexCap', 'ratio','indexWeight','shares','indexMarketValue']]
        sector_frames.append(dataForSql)


    sectorIndexData = pd.concat(sector_frames, axis=0)
    sectorIndexData.to_sql(con=cnxn, if_exists='replace', name = 'EqSectorIx', index = False)

    divisorDf = pd.DataFrame.from_dict(indexDivisors, orient = 'index').reset_index()
    divisorDf.columns = ['Sector', 'IndexDivisor']
    divisorDf.to_sql(con=cnxn, if_exists='replace', name = 'EqSectorIxDivisors', index = False)


# # Calculate Index value on a specific date & return from initial calc

# In[5]:


def calculate_index_by_date(constituents, date, sector):
    
    prices = nasdaqdatalink.get_table(Sharadar.PRICES.value, date = date, paginate=True)
        
    prices = prices[prices.ticker.isin(constituents.ticker.tolist())]
    
    prices = prices.set_index('ticker').merge(constituents.set_index('ticker')[['shares']], left_index = True, right_index=True).reset_index()
    
    prices['indexMarketValue'] = prices.closeadj * prices.shares
    
    ixDivisors = pd.read_sql(f"select * from EqSectorIxDivisors where Sector = '{sector}' ", cnxn)

    return prices.indexMarketValue.sum() / ixDivisors['IndexDivisor'].iloc[0]


# In[6]:


sectors = df.sector.unique().tolist()

print(sectors)

frames = []

# dates = [x.strftime('%Y-%m-%d') for x in pd.date_range(start='10/01/2022', end='11/03/2022')]

dates = [prev_bd]

for date in dates:
    perfDict = {}

    for sector in sectors:

        ixdata = pd.read_sql('select * from EqSectorIx', cnxn)
        
        constituents = ixdata[ixdata.sector == sector] 

        indexValue = calculate_index_by_date(constituents, sector=sector, date = date)

        # indexReturn = ((indexValue / 100) -1)  # 100 is the base (starting) value for all index's

        perfDict[sector] = indexValue # store the index value on each date and calculate the return downstream
    
    frames.append(pd.DataFrame.from_dict(perfDict, orient = 'index'))

indexPerformance = pd.concat(frames, axis=1)

indexPerformance.columns = dates

indexPerformance.T.reset_index().to_sql(con=cnxn, if_exists='append', name = 'EqSectorIxPerf', index = False)
# indexPerformance.T


# In[ ]:





# In[7]:


data = pd.read_sql(f"select * from EqSectorIxPerf", cnxn)
data


# In[ ]:




