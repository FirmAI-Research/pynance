p = '/Users/sands/Desktop/LOBSTER_SampleFile_AMZN_2012-06-21_10/'

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
from itertools import chain

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


x = list(chain(*[('Ask Price {0},Ask Size {0},Bid Price {0},Bid Size {0}'.format(i)).split(',') for i in range(10)]))
print(x)


price = list(chain(*[('Ask Price {0},Bid Price {0}'.format(i)).split(',') for i in range(10)]))
size = list(chain(*[('Ask Size {0},Bid Size {0}'.format(i)).split(',') for i in range(10)]))
cols = list(chain(*zip(price, size)))

order_data = 'AMZN_2012-06-21_34200000_57600000_orderbook_10.csv'
orders = pd.read_csv(p + order_data, header=None, names=cols)
print(orders)

orders.info()
print(orders.head())

'''

Message Type Codes:

1: Submission of a new limit order
2: Cancellation (Partial deletion 
   of a limit order)
3: Deletion (Total deletion of a limit order)
4: Execution of a visible limit order                
5: Execution of a hidden limit order
7: Trading halt indicator                  
   (Detailed information below)
'''
types = {1: 'submission',
         2: 'cancellation',
         3: 'deletion',
         4: 'execution_visible',
         5: 'execution_hidden',
         7: 'trading_halt'}


trading_date = '2012-06-21'
levels = 10



message_data = 'AMZN_{}_34200000_57600000_message_{}.csv'.format(
    trading_date, levels)
messages = pd.read_csv(p + message_data,
                       header=None,
                       names=['time', 'type', 'order_id', 'size', 'price', 'direction'])
messages.info()
print(messages.head())

messages.type.map(types).value_counts()
messages.time = pd.to_timedelta(messages.time, unit='s')
messages['trading_date'] = pd.to_datetime(trading_date)
messages.time = messages.trading_date.add(messages.time)
messages.drop('trading_date', axis=1, inplace=True)
messages.head()
data = pd.concat([messages, orders], axis=1)
data.info()
ex = data[data.type.isin([4, 5])]
print(ex.head())

# visible or hidden
cmaps = {'Bid': 'Blues','Ask': 'Reds'}
fig, ax=plt.subplots(figsize=(14, 8))
time = ex['time'].dt.to_pydatetime()
for i in range(10):
    for t in ['Bid', 'Ask']:
        y, c = ex['{} Price {}'.format(t, i)], ex['{} Size {}'.format(t, i)]
        ax.scatter(x=time, y=y, c=c, cmap=cmaps[t], s=1, vmin=1, vmax=c.quantile(.95))
ax.set_xlim(datetime(2012, 6, 21, 9, 30), datetime(2012, 6, 21, 16, 0))
sns.despine()
fig.tight_layout();
plt.show()
