# Calendar Call Strategy

import p4f
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

s0 = 187 # Initial stock price on May 2nd, 2015
k = 190 # Strike price of the May 26th 2015 and July 28th 2015 Call Option
cs = 7.50; # Premium of the Short call on May 2nd, 2015
cl = 20.70; # Premium of the Long call on May 2nd, 2015
shares = 100 # Shares per lot 
sT = np.arange(0,2*s0,5) # Stock Price at expiration of the Call
sigma = 0.4 # Historical Volatility
r = 0.08 # Risk-free interest rate

t = datetime(2015,7,28) - datetime(2015,5,26) ; T = t.days / 365;
# Payoff from the May 26th 2015 Short Call Option at Expiration
y1 = np.where(sT > k,((k - sT) + cs) * shares, cs * shares)
# Value of the July 28th 2015 long Call Option on May 26th 2015
lc_value = p4f.bs_call(sT,k,T,r,sigma) * shares
# Payoff from the Calendar Call
y2 = y1 + (lc_value - cl)
# Create a plot using matplotlib 
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False) # Top border removed 
ax.spines['right'].set_visible(False) # Right border removed
ax.spines['bottom'].set_position('zero') # Sets the X-axis in the center
ax.tick_params(top=False, right=False) # Removes the tick-marks on the RHS

plt.plot(sT,y1,lw=1.5,label='Short Call')
plt.plot(sT,lc_value,lw=1.5,label='Long Call')
plt.plot(sT,y2,lw=1.5,label='Calendar Call')

plt.title('Calendar Call') 
plt.xlabel('Stock Prices')
plt.ylabel('Profit/loss')

plt.grid(True)
plt.axis('tight')
plt.legend(loc=0)
plt.show()