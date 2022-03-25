import numpy as np 
import matplotlib.pyplot as plt
import seaborn 


def call_payoff(sT, strike_price, premium):
    return np.where(sT > strike_price, sT - strike_price, 0) - premium

def put_payoff(sT, strike_price, premium):
    return np.where(sT < strike_price, strike_price - sT, 0) - premium 

def calc_payoff(s0, k_long_put, k_long_call, c_long_put, c_long_call, sT): 
    payoff_long_put = put_payoff(sT, k_long_put, c_long_put)
    payoff_long_call = call_payoff(sT, k_long_call, c_long_call)
    payoff_straddle = payoff_long_call + payoff_long_put
    return  [payoff_long_put,payoff_long_call,payoff_straddle]
