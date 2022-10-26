#!/usr/bin/env python
# coding: utf-8

# In[1]:


from enum import Enum
import sys, os
import nasdaqdatalink
import pandas as pd 
import numpy as np
from scipy.stats.mstats import gmean

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


# 
# Two tables: One flat; One tall<br><br>
#     CompFunBase - Company Fundamentals with a record indexed by Ticker and Calendar Date<br><br>
#     CompFunRanks - Tall table indexed by Ticker, Industry, and Calendar Date with the companies ranks against their peer group (industry) <br>   Use this melted form to pivot and present ranks as a time series<br>
# <br>
# CompFunIndStats - descriptive statistics at the industry level for each date and metric
# <br><br>
# https://data.nasdaq.com/api/v3/datatables/SHARADAR/SF1?qopts.export=true&api_key=API_KEY
# 
# 

# # Join equity fundamentals and static company info

# In[ ]:


# Fundamentals
fun = Fundamentals()
df_fun = fun.full_export(curl = False) # Set curl = True if data should be refreshed
df_fun = df_fun[df_fun.dimension == 'MRQ']

# Static Profile info
nasdaq.Nasdaq()
tick = nasdaq.Tickers()
df_prof = tick.full_export(curl = False) # Set curl = True if data should be refreshed

#Industry list
all_industies = df_prof.industry.unique().tolist()
# print(all_industies)

# Join
df = df_fun.set_index('ticker').merge(df_prof.set_index('ticker'), how='inner', left_index=True, right_index=True).reset_index()
df


# # Subset for rank columns of interest

# In[ ]:


df = df[Columns.RANKS.value]


# In[ ]:


cols = ['ticker', 'calendardate'] + [c for c in df.columns if c not in ['ticker', 'calendardate']]
df = df[cols]


# In[ ]:


df.shape


# In[ ]:


# df.columns.tolist()


# # 1. load fundamentals to flat table

# In[ ]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
cnxn = engine.connect()
df.to_sql(con=cnxn, if_exists='replace', name = 'CompFunBase', index = False) #Company Fundamentals base


# In[ ]:


base = pd.read_sql("select * from CompFunBase where industry = 'Utilities - Regulated Electric' and ticker = 'DUK'", cnxn)
base


# # 2. load ranks to tall table

# CompFunRanks

# In[ ]:


dates = df['calendardate'].unique().tolist()
industries =  df['industry'].unique().tolist()


# In[ ]:


frames =[]
for date in dates[-6:]:
    for industry in industries:
        data = df[(df.calendardate == date) & (df.industry == industry)].set_index(['ticker','calendardate','industry'])
        ranks = data.rank(axis=1, pct=True, numeric_only = True).reset_index()
        melt = ranks.melt(id_vars = ['ticker', 'calendardate','industry'])
        frames.append(melt)
res = pd.concat(frames, axis=0)


# In[ ]:


res


# In[ ]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
cnxn = engine.connect()
res.to_sql(con=cnxn, if_exists='replace', name = 'CompFunRanks', index = False) #Company Fundamentals Ranks


# In[ ]:


# res[(res.ticker == 'AMZN') & (res.variable == 'revenue')]


# In[ ]:


rank = pd.read_sql("select * from CompFunRanks where ticker == 'AMZN'", cnxn)
rank


# # 3. Industry descriptive stats

# In[ ]:


dates = df['calendardate'].unique().tolist()
industries =  df['industry'].unique().tolist()


# In[ ]:


frames =[]
for date in dates[-6:]:
    for industry in industries:
        data = df[(df.calendardate == date) & (df.industry == industry)].set_index(['ticker','calendardate','industry'])
        stats = data.describe().reset_index()
        stats.rename(columns={'index':'stat'}, inplace = True)
        stats['calendardate'] = date
        stats['industry'] = industry
        melt = stats.melt(id_vars = ['calendardate','industry','stat'])
        frames.append(melt)
res = pd.concat(frames, axis=0)


# In[ ]:


res


# In[ ]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
cnxn = engine.connect()
res.to_sql(con=cnxn, if_exists='replace', name = 'CompFunIndStats', index = False) #Company Fundamentals Ranks


# In[ ]:




