import numpy as np
import matplotlib.pyplot as plt

s0 = 225 # Initial stock price 
k1 = 215;c1 = 12.50; # Strike & premium for ITM Call
k2 = 225;c2 = 6.50; # Strike & premium for ATM Call
k3 = 235;c3 = 3.00; # Strike & premium for OTM Call
shares = 100 # Shares per lot
# Stock Price at expiration of the Call
sT = np.arange(0,2*s0,5)
# Payoff from the Lower Strike ITM Long Call Option
y1 = np.where(sT > k1,((sT-k1) - c1) * shares, -c1 * shares)
# Payoff from ATM Short Call Option
y2 = np.where(sT > k2,((k2-sT) + c2) * 2 * shares, c2 * 2 * shares )
# Payoff from the Higher Strike OTM Long Call Option
y3 = np.where(sT > k3,((sT-k3) - c3) * shares, -c3 * shares)
# Payoff for the Long Call Butterfly
y = y1 + y2 + y3
# Create a plot using matplotlib 
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False) # Top border removed 
ax.spines['right'].set_visible(False) # Right border removed
ax.spines['bottom'].set_position('zero') # Sets the X-axis in the center
ax.tick_params(top=False, right=False) # Removes the tick-marks on the RHS

plt.plot(sT,y,lw=1.5)

plt.title('Long Call Butterfly') 
plt.xlabel('Stock Prices')
plt.ylabel('Profit/loss')

plt.grid(True)
plt.axis('tight')
plt.show()