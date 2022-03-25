import numpy as np
import matplotlib.pyplot as plt
import seaborn
from tabulate import tabulate
seaborn.set(style='darkgrid')


def call_payoff(sT, strike_price, premium):
    return np.where(sT > strike_price, sT - strike_price, 0)-premium



s0 = 1860
strike_price_long_call = 1880
premium_long_call = 16.15
strike_price_short_call = 1860
premium_short_call = 23.8
sT = np.arange(1800,1920,10)


long_call_payoff = call_payoff(sT, strike_price_long_call, premium_long_call )

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, long_call_payoff, color='g')
ax.set_title('LONG 1880 Strike Call')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')

plt.show()



short_call_payoff = call_payoff(sT, strike_price_short_call, premium_short_call )*-1.0

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, short_call_payoff, color='r')
ax.set_title('Short 1860 Strike Call')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')

plt.show()









def put_payoff(sT, strike_price, premium):
    return np.where(sT < strike_price, strike_price - sT, 0) - premium
s0 = 1860
strike_price_long_put =1840
premium_long_put = 17
strike_price_short_put = 1860
premium_short_put = 25.5
sT = np.arange(1800,1920,10)


long_put_payoff = put_payoff(sT, strike_price_long_put, premium_long_put)

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, long_put_payoff, color ='g')
ax.set_title('Long 1840 Strike Put')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()



short_put_payoff = put_payoff(sT, strike_price_short_put, premium_short_put)*-1.0

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, short_put_payoff, color ='r')
ax.set_title('Short 1860 Strike Put')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()



Iron_Butterfly_payoff = long_call_payoff + short_call_payoff + long_put_payoff + short_put_payoff 

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT, Iron_Butterfly_payoff, color ='b')
ax.set_title('Iron Butterfly Spread')
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()




profit = max (Iron_Butterfly_payoff)
loss = min (Iron_Butterfly_payoff)
print ("Max Profit %.2f" %profit)
print ("Min Loss %.2f" %loss)

fig, ax = plt.subplots(figsize=(10,5))
ax.spines['bottom'].set_position('zero')
ax.plot(sT, Iron_Butterfly_payoff, color ='b', label ='Iron Butterfly Spread')
ax.plot(sT, long_call_payoff,'--', color ='g', label = 'Long Call')
ax.plot(sT, short_put_payoff,'--', color ='r', label = 'Short Call')
ax.plot(sT, long_put_payoff,'--', color ='g',label = 'Long Put')
ax.plot(sT, short_put_payoff,'--', color ='r',label = 'Short Put')
plt.legend()
plt.xlabel('Stock Price (sT)')
plt.ylabel('Profit & Loss')
plt.show()