import numpy as np
import matplotlib.pyplot as plt
import seaborn


def call_payoff(sT, strike_price, premium):
return np.where(sT > strike_price, sT - strike_price, 0) – premium
spot_price = 323.40
strike_price_long_call = 370
premium_long_call = 1.30
strike_price_short_call = 350
premium_short_call = 3.30
sT = np.arange(0.5*spot_price,2*spot_price,1)

payoff_long_call = call_payoff(sT, strike_price_long_call, premium_long_call)

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,label='Long 370 Strike Call',color='g')
plt.xlabel('Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
#ax.spines['top'].set_visible(False) # Top border removed
#ax.spines['right'].set_visible(False) # Right border removed
#ax.tick_params(top=False, right=False) # Removes the tick-marks on the RHS
plt.grid()
plt.show()



payoff_short_call = call_payoff(sT, strike_price_short_call, premium_short_call) * -1.0

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_short_call,label='Short 350 Strike Call',color='r')
plt.xlabel('Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.grid()
plt.show()



def put_payoff(sT, strike_price, premium):
return np.where(sT < strike_price, strike_price - sT, 0) – premium
spot_price = 323.40
strike_price_long_put = 280
premium_long_put = 1.20
strike_price_short_put = 300
premium_short_put = 3.40
sT = np.arange(0.5*spot_price,2*spot_price,1)

payoff_long_put = put_payoff(sT, strike_price_long_put, premium_long_put)

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_put,label='Long 280 Strike Put',color='y')
plt.xlabel('Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.grid()
plt.show()



payoff_short_put = put_payoff(sT, strike_price_short_put, premium_short_put) * -1.0

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_short_put,label='Short 300 Strike Put',color='m')
plt.xlabel('Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.grid()
plt.show()




payoff = payoff_long_call + payoff_short_call + payoff_long_put + payoff_short_put

fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,'--',label='Long 370 Strike Call',color='g')
ax.plot(sT,payoff_short_call,'--',label='Short 350 Strike Call',color='r')
ax.plot(sT,payoff_long_put,'--',label='Long 280 Strike Put',color='y')
ax.plot(sT,payoff_short_put,'--',label='Short 300 Strike Put',color='m')
ax.plot(sT,payoff,label='Iron Condor')
plt.xlabel('Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.grid()
plt.show()