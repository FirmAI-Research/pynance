from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

import configparser
c = configparser.ConfigParser()
c.read('../pyalgo.cfg')

import quandl as q
#q.ApiConfig.api_key = c['quandl']['-sZL9MEeYDAGRHMhb7Uz']
q.ApiConfig.api_key = '-sZL9MEeYDAGRHMhb7Uz'
d = q.get('BCHAIN/MKPRU')
d['SMA'] = d['Value'].rolling(100).mean()
d.loc['2013-1-1':].plot(title='BTC/USD exchange rate',
                        figsize=(10, 6));
# plt.savefig('../../images/ch01/bitcoin_xr.png')

print(d.tail())


