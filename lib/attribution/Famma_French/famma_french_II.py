 # famma french from pakt cookbook
from numpy import isin
import pandas_datareader.data as web
import pandas as pd 
import yfinance as yf
import sys
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from lib.calendar import Calendar
cal = Calendar()

cwd = os.getcwd()
img_dirp = os.path.join(cwd, 'attribution/static')
print(img_dirp)


class FammaFrench():
    """[Summary]
    
    :param [ParamName]: 
    
    :raises [ErrorType]: 
    """    

    def __init__(self, symbols, weights) -> None:
        self.symbols = symbols
        self.weights = weights

        self.START_DATE = '2014-01-01'
        
        # run
        self.ff_data = self.get_ff_factors()
        self.asset_returns = self.get_returns()
        self.data = self.merge_factors_and_portfolio()
        self.model_summary = self.three_factor()


    def get_ff_factors(self, n=3):
        ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                start=self.START_DATE)    
        print(ff_dict['DESCR'])        
        df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                        start=self.START_DATE)[0]
        df_three_factor = df_three_factor.div(100)
        df_three_factor.index = df_three_factor.index.strftime('%Y-%m-%d')
        df_three_factor.index.name = 'Date'  
        df_three_factor.index = pd.to_datetime(df_three_factor.index)
        return df_three_factor


    def get_returns(self):
        ''' calculate weighted returns of a portfolio'''
        frames = []
        print(len(self.symbols))
        if isinstance(self.symbols, list) and len(self.symbols) > 1:
            print('Using weighting scheme')
            for symbol in self.symbols:
                returns = pd.DataFrame(yf.download(symbol,self.START_DATE, cal.today())['Adj Close'].pct_change().dropna())
                returns.rename(columns = {'Adj Close':symbol}, inplace = True)
                frames.append(returns)
            from functools import reduce
            df = reduce(lambda x, y: pd.merge(x, y, on = 'Date'), frames)
            df['portf_rtn'] = pd.DataFrame(df.dot(self.weights)) # calculate weighted returns via matrix multiplication
            df = df[[x for x in df.columns if x not in self.symbols]] # drop individual asset returns and keep only weighted portfolio returns

        else:
            print('Not using weighting scheme')
            symbol = self.symbols[0]
            returns = pd.DataFrame(yf.download(symbol,self.START_DATE, cal.today())['Adj Close'].pct_change().dropna())
            returns.rename(columns = {'Adj Close':'portf_rtn'}, inplace = True)
            df = returns
        df.index = pd.to_datetime(df.index)
        df = df.resample('M').sum()
        return df


    def merge_factors_and_portfolio(self, download_ff_data=True):
        return  self.ff_data.merge(self.asset_returns,  left_index = True, right_index = True, how='inner')


    def rolling_factor_model(self, input_data, formula, window_size):
        '''
        Function for estimating the Fama-French (n-factor) model using a rolling window of fixed size.
        
        Parameters
        ------------
        input_data : pd.DataFrame
            A DataFrame containing the factors and asset/portfolio returns
        formula : str
            `statsmodels` compatible formula representing the OLS regression  
        window_size : int
            Rolling window length.
        
        Returns
        -----------
        coeffs_df : pd.DataFrame
            DataFrame containing the intercept and the three factors for each iteration.
        '''

        coeffs = []

        for start_index in range(len(input_data) - window_size + 1):        
            end_index = start_index + window_size

            # define and fit the regression model 
            ff_model = smf.ols(
                formula=formula, 
                data=input_data[start_index:end_index]
            ).fit()
    
            # store coefficients
            coeffs.append(ff_model.params)
        
        coeffs_df = pd.DataFrame(
            coeffs, 
            index=input_data.index[window_size - 1:]
        )

        return coeffs_df


    def three_factor(self):
          
        plt.rcParams["figure.figsize"] = (20,7)

        ff_data = self.data
        print(ff_data)

        ff_data.columns = [ 'mkt', 'smb', 'hml', 'rf', 'portf_rtn']
        ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf
        
        ff_model = smf.ols(formula='portf_ex_rtn ~ mkt + smb + hml',
        data=ff_data).fit()
        print(f'Results for: {" ".join(self.symbols)}')
        print(ff_model.summary())

        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])
        MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
        results_df = self.rolling_factor_model(ff_data, 
                                        MODEL_FORMULA, 
                                        window_size=12)
        results_df.plot(title = f'{self.symbols} - Rolling Fama-French Three-Factor model')
        plt.savefig(os.path.join(img_dirp, 'img/ff_performance_attribution.png'))

        return ff_model.summary()
    




    ''' multiple regression models '''

    # def three_factor(self):
    #     self.model = smf.formula.ols(
    #         formula="port_excess ~ mkt_excess + SMB + HML", data=self.df).fit()

    # def four_factor(self):
    #     self.model = smf.formula.ols(
    #         formula="port_excess ~ mkt_excess + SMB + HML + Mom", data=self.df).fit()

    # def five_factor(self):
    #     self.model = smf.formula.ols(
    #         formula="port_excess ~ mkt_excess + SMB + HML + ST_Rev + LT_Rev + Mom", data=self.df).fit()

    # def print_summary(self):
    #     print(self.df.tail())
    #     print(self.model.summary())
    #     print('Parameters: ', self.model.params)
    #     print('R2: ', self.model.rsquared)

    # def plot(self):
    #     ((self.df + 1).cumprod()).plot(figsize=(15, 7))
    #     plt.title(f"Famma French Factors", fontsize=16)
    #     plt.ylabel('Portfolio Returns', fontsize=14)
    #     plt.xlabel('Year', fontsize=14)
    #     plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    #     plt.legend()
    #     plt.yscale('log')
    #     plt.show()

    # def plot_text(self):
    #     plt.text(0.01, 0.05, str(self.model.summary()), {
    #              'fontsize': 10}, fontproperties='monospace')



    # def calc_betas():
    #     bm = calc_bm
    #     subadvisor_strs = [x for x in merge.columns[3:]]
    #     betas = []
    #     for subadv in subadvisor_strs:
    #         covariance = merge[[subadv,calc_bm]].cov().iloc[0,1]
    #         benchmark_variance = merge[[calc_bm]].var()
    #         beta = covariance / benchmark_variance
    #         betas.append(beta)
    #     x = pd.DataFrame(pd.concat(betas)).transpose()
    #     x.columns =  subadvisor_strs
    #     return x



    # def capm():
    #     _merge = merge.copy()
    #     for subadv in subadvisor_strs:

    #         # separate target
    #         y = _merge.pop(subadv)

    #         # add constant
    #         X = sm.add_constant(_merge[calc_bm])

    #         # define and fit the regression model 
    #         capm_model = sm.OLS(y, X).fit()

            
    #         y = merge[subadv]                 # dependant variable
    #         x = merge[calc_bm]                 # independent variables  
    #         # print results 
    #         print(capm_model.summary())
    #         # results = model.fit()
    #         plt.figure(figsize=(20,5))
    #         plt.scatter(y,y,alpha=0.3)
    #         y_true = y_predict = capm_model.params[0] + capm_model.params[1]*x
    #         plt.plot(x, y_predict, linewidth=3)
    #         plt.xlabel(calc_bm)
    #         plt.ylabel(subadv)
    #         plt.legend()
    #         plt.title('OLS Regression')
    #         plt.show()


     
    # def sensitivity_analysis():
    #     import numpy as np
    #     import pandas as pd
    #     import matplotlib.pyplot as plt
    #     import statsmodels.api as sm

    #     frames = []
    #     for y_str in [c for c in returns.columns if c != 'Date']:

    #         y = merge[y_str]                      
    #         x = merge[calc_bm]                  

    #         X = sm.add_constant(x)
    #         model = sm.OLS(y,X).fit()

    #         x1n = np.linspace(-5, 5, 10)
    #         Xnew = sm.add_constant(x1n)
    #         ynewpred = model.predict(Xnew)
            
    #         res = pd.DataFrame([x1n, ynewpred]).transpose().rename(columns={0:calc_bm, 1:y_str})
    #         res.set_index(calc_bm, inplace=True)
    #         frames.append(res)

    #     import pandas as pd
    #     from functools import reduce
    #     df = reduce(lambda x, y: pd.merge(x, y, on = calc_bm), frames)
    #     df

