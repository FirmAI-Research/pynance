import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import yfinance as yf
from datetime import date
import pandas_datareader.data as web
from lib.calendar import Calendar
cal = Calendar()

today = date.today()

import sys, os
cwd = os.getcwd()
fp = (os.path.join(os.path.dirname(cwd), 'lib', 'attribution','PCA', 'portfolio_sensitivity_to_ir.csv'))


class PCA_VaR:
    '''
    meausre risk of a portfolio using PCA

    https://abhyankar-ameya.medium.com/exploring-risk-analytics-using-pca-with-python-3aca369cbfe4
    '''

    def __init__(self) -> None:
        
        tr = self.get_treasury_rates()
        print(tr)

        x = tr.copy().reset_index().dropna()
        df = x.drop(axis=1,columns=['DATE'])
        X = df.values
        self.X = scale(X)        #Normalization of the data

        self.df_factor_loading, self.variance_percent_df, self.variance_ratio_df = self.compute_factors()

        self.portfolio_data = pd.read_csv(fp)

        self.build_results()

    def get_treasury_rates(self):
        frames = []
        for ticker in ['TB3MS', 'TB6MS', 'DGS1', 'DGS2', 'DGS3', 'DGS5', 'DGS7',  'DGS10', 'DGS30']:
            stockdata = web.DataReader(ticker, 'fred', pd.to_datetime('2010-01-01'), cal.today())
            frames.append(stockdata)
        return pd.concat(frames, axis=1)


    def historical_prices_cbind(self, arr= [],  date_start = pd.to_datetime('2010-01-01'), date_end =cal.today()):
        if date_end==None:
            date_end = pd.datetime.now().date()
            print(date_end)
        df = yf.download(arr,date_start)['Adj Close']
        print(df.head())
        return df

    def compute_factors(self):
        '''
        Factor loadings explain the relation between the impact of a factor on interest rates at respective tenor points.
        we will see which PC contributes how much amount of variance/dispersion.

        i.e. From the table alongside, we observe that PC1 explains almost 96% of the total variation, and PC2 explains close to 1.95% of total variation. Therefore, rather than using all PCs in the subsequent calculation, we will only use PC1 and PC2 in further calculation as these two components explain close to 98% of the total variance.

        1. PC1 corresponds to the roughly the parallel shift in the yield curve.
        2. PC2 corresponds to roughly a steepening in the yield curve.
        
        This is in-line with the theory of fixed income risk measurement which states that majority of the movement in the price of a bond is explained by the parallel shift in the yield curve and the residual movements in the price is explained by steepening and curvature of the interest rate curve
        '''
        pca = PCA(n_components=3)
        pca.fit(self.X)
        factor_loading = pca.components_
        df_factor_loading = pd.DataFrame(factor_loading)
        variance_percent_df = pd.DataFrame(data=pca.explained_variance_)
        variance_ratio_df = pd.DataFrame(data=pca.explained_variance_ratio_)
        variance_ratio_df = variance_ratio_df * 100
        return df_factor_loading, variance_percent_df, variance_ratio_df


    def build_results(self):
        '''
        calculating VaR using PCA
        the respective factor loading (from step ‘c’ above) is multiplied by the PV01 for that specific tenor (from step ‘d’ above). This result is then multiplied by the variance of the factor loadings (from step ‘c’ above). We will do this for both the PCs namely PC1 and PC2 .
        '''
        df_calculation = pd.DataFrame()
        df_calculation['PortfolioData'] = self.portfolio_data.iloc[:,1]
        df_calculation['PC1'] = self.df_factor_loading.iloc[:,0]
        df_calculation['PC2'] = self.df_factor_loading.iloc[:,1]

        df_calculation['Result1'] = df_calculation['PC1'] * df_calculation['PortfolioData']
        df_calculation['Result2'] = df_calculation['PC2'] * df_calculation['PortfolioData']
        print(df_calculation)
        result1 = ((df_calculation['Result1'].sum())**2) * self.variance_percent_df.iloc[0]
        result2 = ((df_calculation['Result2'].sum())**2) * self.variance_percent_df.iloc[1]

        portfolio_risk = np.sqrt(result1+result2) * 2.33 #99 percentile value
        print('portfolio risk:', portfolio_risk)




class VaR:
    '''
        https://shankarmsy.github.io/posts/pca-sklearn.html
    '''

    def __init__(self) -> None:
        pass








# stocks_ = historical_prices_cbind(['AMZN', 'BLK', 'JPM', 'NEE'],'1975-01-01',today).dropna() 
# print(stocks_)
# # stocks_ = pd.read_csv('C:\indu_dly.csv',index_col =0)
# indu_index = yf.download('SPY','1975-01-01',today)['Adj Close']

# # get cumulative total return index normalized at 1
# indu_index_ret = indu_index.pct_change()
# indu_ret_idx = indu_index_ret[1:]+1
# indu_ret_idx= indu_ret_idx.cumprod()
# indu_ret_idx.columns =['dow']

# # calculate stock covariance
# stocks_ret = stocks_.pct_change()
# stocks_ret = stocks_ret[1:] # skip the first row (NaN)
# stocks_cov = stocks_ret.cov()

# # USING SKLEARN
# ncomps = 2
# sklearn_pca = sklearnPCA(n_components=ncomps) # let's look at the first 20 components
# pc = sklearn_pca.fit_transform(stocks_ret)


# factor_loading = sklearn_pca.components_
# df_factor_loading = pd.DataFrame(factor_loading)
# print(df_factor_loading)
# print("Meaning of the components:")
# for component in sklearn_pca.components_: 
#     print(" + ".join("%.2f x %s" % (value, name) for value, name in zip(component, stocks_.columns)))




# # plot the variance explained by pcs
# plt.bar(range(ncomps),sklearn_pca.explained_variance_ratio_)
# plt.title('variance explained by pc')
# plt.show()

# print(sklearn_pca.explained_variance_ratio_)


# # get the Principal components
# pcs =sklearn_pca.components_
# print(pcs)


# # first component
# pc1 = pcs[0,:]
# # normalized to 1 
# pc_w = np.asmatrix(pc1/sum(pc1)).T

# # apply our first componenet as weight of the stocks
# pc1_ret = stocks_ret.values*pc_w

# # plot the total return index of the first PC portfolio
# pc_ret = pd.DataFrame(data =pc1_ret, index= stocks_ret.index)
# pc_ret_idx = pc_ret+1
# pc_ret_idx= pc_ret_idx.cumprod()
# pc_ret_idx.columns =['pc1']

# pc_ret_idx['indu'] = indu_index[1:]
# pc_ret_idx.plot(subplots=True,title ='PC portfolio vs Market',layout =[1,2])
# plt.show()

# # plot the weights in the PC
# weights_df = pd.DataFrame(data = pc_w*100,index = stocks_.columns)
# weights_df.columns=['weights']
# weights_df.plot.bar(title='PCA portfolio weights',rot =45,fontsize =8)
# plt.show()


