import numpy as np
import matplotlib.pyplot as plt
import seaborn

def call_payoff(sT, strike_price, premium):
return np.where(sT > strike_price, sT - strike_price, 0) – premium



spot_price = 1130
strike_price_long_call = 1160
premium_long_call = 20
strike_price_short_call = 1200
premium_short_call = 11
sT = np.arange(0.95*spot_price,1.1*spot_price,1)


payoff_long_call = call_payoff(sT, strike_price_long_call, premium_long_call)
# Plot
fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,label='Long 1160 Strike Call',color='g')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.show()


payoff_short_call = call_payoff(sT, strike_price_short_call, premium_short_call) * -1.0
# Plot
fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_short_call,label='Short 1200 Strike Call',color='r')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.show()


payoff_bull_call_spread = payoff_long_call + payoff_short_call

print "Max Profit:", max(payoff_bull_call_spread)
print "Max Loss:", min(payoff_bull_call_spread)
# Plot
fig, ax = plt.subplots()
ax.spines['bottom'].set_position('zero')
ax.plot(sT,payoff_long_call,'--',label='Long 1160 Strike Call',color='g')
ax.plot(sT,payoff_short_call,'--',label='Short 1200 Strike Call ',color='r')
ax.plot(sT,payoff_bull_call_spread,label='Bull Call Spread')
plt.xlabel('Infosys Stock Price')
plt.ylabel('Profit and loss')
plt.legend()
plt.show()