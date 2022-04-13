import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

import yfinance as yf
from datetime import date

today = date.today()
# read in data
def historical_prices_cbind(arr= [],  date_start = '', date_end =None):
    """
    column bind time series of historical stock returns for an array of individual securities
    _________
    :param arr: array of individual tickers to retrieve historical prices for
    :type list: 
    :param plot: line chart of all columns in df   
    :type bool: 
    :param date_start: date to start retrieval of data
    :type pd.datetime: 
    :param date_end: date to end retrieval of data
    :type pd.datetime:  
    ----------
    :return: timeseries of historical prices
    :rtype: pd.dataframe
    _________
    """
    if date_end==None:
        date_end = pd.datetime.now().date()
        print(date_end)

    df = yf.download(arr,date_start)['Adj Close']
    print(df.head())

    return df

stocks_ = historical_prices_cbind(['AAPL','JNJ','BABA','NLY','VZ','AMT','T','MLM'],'1975-01-01',today).dropna() 
print(stocks_)
# stocks_ = pd.read_csv('C:\indu_dly.csv',index_col =0)
indu_index = yf.download('SPY','1975-01-01',today)['Adj Close']

# get cumulative total return index normalized at 1
indu_index_ret = indu_index.pct_change()
indu_ret_idx = indu_index_ret[1:]+1
indu_ret_idx= indu_ret_idx.cumprod()
indu_ret_idx.columns =['dow']

# calculate stock covariance
stocks_ret = stocks_.pct_change()
stocks_ret = stocks_ret[1:] # skip the first row (NaN)
stocks_cov = stocks_ret.cov()

# USING SKLEARN
ncomps = 4
sklearn_pca = sklearnPCA(n_components=ncomps) # let's look at the first 20 components
pc = sklearn_pca.fit_transform(stocks_ret)


factor_loading = sklearn_pca.components_
df_factor_loading = pd.DataFrame(factor_loading)
print(df_factor_loading)
print("Meaning of the components:")
for component in sklearn_pca.components_: 
    print(" + ".join("%.2f x %s" % (value, name) for value, name in zip(component, stocks_.columns)))




# plot the variance explained by pcs
plt.bar(range(ncomps),sklearn_pca.explained_variance_ratio_)
plt.title('variance explained by pc')
plt.show()

print(sklearn_pca.explained_variance_ratio_)


# get the Principal components
pcs =sklearn_pca.components_
print(pcs)


# first component
pc1 = pcs[0,:]
# normalized to 1 
pc_w = np.asmatrix(pc1/sum(pc1)).T

# apply our first componenet as weight of the stocks
pc1_ret = stocks_ret.values*pc_w

# plot the total return index of the first PC portfolio
pc_ret = pd.DataFrame(data =pc1_ret, index= stocks_ret.index)
pc_ret_idx = pc_ret+1
pc_ret_idx= pc_ret_idx.cumprod()
pc_ret_idx.columns =['pc1']

pc_ret_idx['indu'] = indu_index[1:]
pc_ret_idx.plot(subplots=True,title ='PC portfolio vs Market',layout =[1,2])
plt.show()

# plot the weights in the PC
weights_df = pd.DataFrame(data = pc_w*100,index = stocks_.columns)
weights_df.columns=['weights']
weights_df.plot.bar(title='PCA portfolio weights',rot =45,fontsize =8)
plt.show()


#https://abhyankar-ameya.medium.com/exploring-risk-analytics-using-pca-with-python-3aca369cbfe4
#https://shankarmsy.github.io/posts/pca-sklearn.html
