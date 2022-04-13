# Standard Python library imports
import pandas as pd

# 3rd party imports
import quandl

def fetch_data(assets):
    
    '''
    Parameters:
    -----------
        assets: list 
            takes in a range of ticker symbols to be pulled from Quandl.  
            Note: It's imporant to make the data retrieval flexible to multiple assets because
            we will likely want to back-test strategies based on cross-asset technical 
            indictors are ratios!  It's always a good idea to put the work and thought 
            in upfront so that your functions are as useful as possible.
            
    Returns:
    --------
        df: Dataframe 
            merged market data from Quandl using the date as the primary merge key.
    '''
        
    count = 0
    auth_tok = '-sZL9MEeYDAGRHMhb7Uz' # quandl api key
    
    #instatiate an empty dataframe
    df = pd.DataFrame()
    
    # Because we will potentially be merging multiple tickers, we want to rename
    # all of the income column names so that they can be identified by their ticker name.  
    for asset in assets:  
        if (count == 0):
            df = quandl.get('EOD/' + asset, authtoken = auth_tok)
            column_names=list(df)

            for i in range(len(column_names)):
                column_names[i] = asset + '_' + column_names[i]
            df.columns = column_names   
            count += 1      
        else:         
            temp = quandl.get('EOD/' + asset, authtoken = auth_tok) 
            column_names = list(temp)

            for i in range(len(column_names)):
                column_names[i] = asset+'_' + column_names[i]
            
            temp.columns = column_names
            # Merge all the dataframes into one with new column names
            df = pd.merge(df, temp, how = 'outer', left_index = True, right_index = True)
        
    return df.dropna(inplace = True)


data = fetch_data (['TLT','GLD','SPY','QQQ','VWO'])
features = [f for f in list(df) if "Adj_Close" in f]
print(features)
adj_closes = df[features]
list(adj_closes)
import matplotlib.pyplot as plt
(adj_closes/adj_closes.iloc[0]*100).plot(figsize=(18,14))

import numpy as np
returns = np.log(adj_closes/adj_closes.shift(1))
returns.mean()
returns.mean() * 252
returns.cov()
returns.cov() * 252
weights = np.random.dirichlet(np.ones(num_assets), size=1)
weights = weights[0]
print(weights)
exp_port_return = np.sum(returns.mean()*weights)*252
print(exp_port_return)
port_var = np.dot(weights.T, np.dot(returns.cov()*252, weights))
port_vol = np.sqrt(port_var)
print(port_var)
print(port_vol)


# Standard python imports
import time
import numpy as np


def portfolio_simulation(assets, iterations):
    '''
    Runs a simulation by randomly selecting portfolio weights a specified
    number of times (iterations), returns the list of results and plots 
    all the portfolios as well.
    
    Parameters:
    -----------  
        assets: list
            all the assets that are to be pulled from Quandl to comprise
            our portfolio.    
        iterations: int 
            the number of randomly generated portfolios to build.
    
    Returns:
    --------
        port_returns: array
            array of all the simulated portfolio returns.
        port_vols: array
            array of all the simulated portfolio volatilities.
    '''
    
    start = time.time()
    num_assets = len(assets)
    
    # Fetch data    
    df = fetch_data(assets)
    features = [f for f in list(df) if "Adj_Close" in f]
    adj_closes = df[features]
    returns = np.log(adj_closes / adj_closes.shift(1))
    
    port_returns = []
    port_vols = []
    
    for i in range (iterations):
        weights = np.random.dirichlet(np.ones(num_assets),size=1)
        weights = weights[0]
        port_returns.append(np.sum(returns.mean() * weights) * 252)
        port_vols.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))
    
    # Convert lists to arrays
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
 
    
    # Plot the distribution of portfolio returns and volatilities 
    plt.figure(figsize = (18,10))
    plt.scatter(port_vols,port_returns,c = (port_returns / port_vols), marker='o')
    plt.xlabel('Portfolio Volatility')
    plt.ylabel('Portfolio Return')
    plt.colorbar(label = 'Sharpe ratio (not adjusted for short rate)')
    
    print('Elapsed Time: %.2f seconds' % (time.time() - start))
    
    return port_returns, port_vols



assets = ['TLT','GLD','SPY','QQQ','VWO']
port_returns, port_vols = portfolio_simulation(assets, 3000)



def portfolio_stats(weights, returns):
    
    '''
    We can gather the portfolio performance metrics for a specific set of weights.
    This function will be important because we'll want to pass it to an optmization
    function to get the portfolio with the best desired characteristics.
    
    Note: Sharpe ratio here uses a risk-free short rate of 0.
    
    Paramaters: 
    -----------
        weights: array, 
            asset weights in the portfolio.
        returns: dataframe
            a dataframe of returns for each asset in the trial portfolio    
    
    Returns: 
    --------
        dict of portfolio statistics - mean return, volatility, sharp ratio.
    '''

    # Convert to array in case list was passed instead.
    weights = np.array(weights)
    port_return = np.sum(returns.mean() * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe = port_return/port_vol

    return {'return': port_return, 'volatility': port_vol, 'sharpe': sharpe}



def minimize_sharpe(weights, returns):  
    return -portfolio_stats(weights)['sharpe'] 

def minimize_volatility(weights, returns):  
    # Note that we don't return the negative of volatility here because we 
    # want the absolute value of volatility to shrink, unlike sharpe.
    return portfolio_stats(weights)['volatility'] 

def minimize_return(weights, returns): 
    return -portfolio_stats(weights)['return']



constraints = ({'type' : 'eq', 'fun': lambda x: np.sum(x) -1})
bounds = tuple((0,1) for x in range(num_assets))
initializer = num_assets * [1./num_assets,]

print (initializer)
print (bounds)


import scipy.optimize as optimize
optimal_sharpe=optimize.minimize(minimize_sharpe,
                                 initializer,
                                 method = 'SLSQP',
                                 bounds = bounds,
                                 constraints = constraints)
print(optimal_sharpe)


optimal_sharpe_weights=optimal_sharpe['x'].round(4)
list(zip(assets,list(optimal_sharpe_weights)))


optimal_stats = portfolio_stats(optimal_sharpe_weights)
print(optimal_stats)

print('Optimal Portfolio Return: ', round(optimal_stats['return']*100,4))
print('Optimal Portfolio Volatility: ', round(optimal_stats['volatility']*100,4))
print('Optimal Portfolio Sharpe Ratio: ', round(optimal_stats['sharpe'],4))


optimal_variance=optimize.minimize(minimize_volatility,
                                   initializer,
                                   method = 'SLSQP',
                                   bounds = bounds,
                                   constraints = constraints)

print(optimal_variance)
optimal_variance_weights=optimal_variance['x'].round(4)
list(zip(assets,list(optimal_variance_weights)))



# Make an array of 50 returns betweeb the minimum return and maximum return
# discovered earlier.
target_returns = np.linspace(port_returns.min(),port_returns.max(),50)

# Initialize optimization parameters
minimal_volatilities = []
bounds = tuple((0,1) for x in weights)
initializer = num_assets * [1./num_assets,]

for target_return in target_returns:
    
    constraints = ({'type':'eq','fun': lambda x: portfolio_stats(x)['return']-target_return},
                   {'type':'eq','fun': lambda x: np.sum(x)-1})
       
    optimal = optimize.minimize(minimize_volatility,
                              initializer,
                              method = 'SLSQP',
                              bounds = bounds,
                              constraints = constraints)
    
    minimal_volatilities.append(optimal['fun'])

minimal_volatilities = np.array(minimal_volatilities)




import matplotlib.pyplot as plt 

# initialize figure size
plt.figure(figsize=(18,10))

plt.scatter(port_vols,
            port_returns,
            c = (port_returns / port_vols),
            marker = 'o')

plt.scatter(minimal_volatilities,
            target_returns,
            c = (target_returns / minimal_volatilities),
            marker = 'x')

plt.plot(portfolio_stats(optimal_sharpe_weights)['volatility'],
         Portfolio_Stats(optimal_sharpe_weights)['return'],
         'r*',
         markersize = 25.0)

plt.plot(portfolio_stats(optimal_variance_weights)['volatility'],
         portfolio_stats(optimal_variance_weights)['return'],
         'y*',
         markersize = 25.0)

plt.xlabel('Portfolio Volatility')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe ratio (not adjusted for short rate)')



import scipy.interpolate as sci

min_index = np.argmin(minimal_volatilities)
ex_returns = target_returns[min_index:]
ex_volatilities = minimal_volatilities[min_index:]

var = sci.splrep(ex_returns, ex_volatilities)

def func(x):
    # Spline approximation of the efficient frontier
    spline_approx = sci.splev(x,var,der=0)  
    return spline_approx

def d_func(x):
    # first derivative of the approximate efficient frontier function
    deriv = sci.splev(x,var,der=1)
    return deriv

def eqs(p, rfr = 0.01):

    #rfr = risk free rate
    
    eq1 = rfr - p[0]
    eq2 = rfr + p[1] * p[2] - func(p[2])
    eq3=p[1] - d_func(p[2]) 
    return eq1, eq2, eq3

# Initializing the weights can be tricky - I find taking the half-way point between your max return and max
# variance typically yields good results.

rfr = 0.01
m=  port_vols.max() / 2
l = port_returns.max() / 2

optimal = optimize.fsolve(eqs, [rfr,m,l])
print(optimal)



np.round(eqs(optimal),4)



constraints =(
    {'type':'eq','fun': lambda x: portfolio_stats(x)['return']-func(optimal[2])},
    {'type':'eq','fun': lambda x: np.sum(x)-1},
    )

result = optimize.minimize(minimize_volatility,
                           initializer,
                           method = 'SLSQP',
                           bounds = bounds,
                           constraints = constraints)

optimal_weights = result['x'].round(3)

portfolio = list(zip(assets, list(optimal_weights)))
print(portfolio)



#https://kevinvecmanis.io/finance/optimization/2019/04/02/Algorithmic-Portfolio-Optimization.html