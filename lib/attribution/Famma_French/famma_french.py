# famma french from pakt cookbook
import pandas_datareader.data as web
import pandas as pd
import yfinance as yf
import sys
import os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from functools import reduce

from lib.calendar import Calendar
cal = Calendar()

class FammaFrench():

    def __init__(self, symbols, weights) -> None:
        
        self.symbols = symbols
        
        self.weights = weights
        
        self.START_DATE = '2014-01-01'


    def get_returns(self):
        ''' calculate weighted returns of a portfolio'''
        frames, invalid = [], []

        for ix, symbol in enumerate(self.symbols):
            
            returns = pd.DataFrame(yf.download(symbol, self.START_DATE, cal.today(), progress=False)['Adj Close'].pct_change().dropna())
            
            returns.rename(columns={'Adj Close': symbol}, inplace=True)

            frames.append(returns) if len(returns) > 0 else invalid.append(ix) # if no data is returned, note the index of the symbol so the weight can be removed
        
        if len(invalid) > 0:

            for i in invalid:
            
                del self.weights[i]
                
        df = reduce(lambda x, y: pd.merge(x, y, on='Date', how = 'outer'), frames) # Each underlying holding may have a different date range available, so outer join and fillna returns as 0

        df.fillna(0, inplace=True)
        
        df['portf_rtn'] = pd.DataFrame(df.dot(self.weights)) # calculate weighted returns via matrix multiplication

        df = df[[x for x in df.columns if x not in self.symbols]] # drop individual asset returns and keep only weighted portfolio returns

        df.index = pd.to_datetime(df.index)
        
        return df


    def get_ff_three_factor(self):

        df = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=self.START_DATE)[0]
        
        df = df.div(100)
        
        df.index = df.index.strftime('%Y-%m-%d')
        
        df.index.name = 'Date'
        
        df.index = pd.to_datetime(df.index)
        
        return df


    def get_ff_industry_factors(self):
        
        df = web.DataReader('10_Industry_Portfolios', 'famafrench', 
                                        start=self.START_DATE)[0]
        
        df = df.div(100)
        
        df.index = df.index.strftime('%Y-%m-%d')
        
        df.index.name = 'Date'
        
        df.index = pd.to_datetime(df.index)
        return df


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


    def three_factor_model(self, df):

        ff_data = df

        ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
        
        ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf

        ff_model = smf.ols(formula='portf_ex_rtn ~ mkt + smb + hml',
                           data=ff_data).fit()
        
        # print(ff_model.summary())

        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])

        MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
        
        results_df = self.rolling_factor_model(ff_data,
                                               MODEL_FORMULA,
                                               window_size=3)
        
        return ff_model.summary(), results_df


    def industry_factor_model(self, df):

        ff_data = df

        ff_data.columns = ['portf_rtn', 'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth', 'Utils', 'Other']
        
        ff_model = smf.ols(formula='portf_rtn ~ NoDur + Durbl + Manuf + Enrgy + HiTec + Telcm + Shops + Hlth + Utils + Other', data=ff_data).fit()
        
        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])

        MODEL_FORMULA = 'portf_rtn ~ NoDur + Durbl + Manuf + Enrgy + HiTec + Telcm + Shops + Hlth + Utils + Other'
        
        results_df = self.rolling_factor_model(ff_data, 
                                          MODEL_FORMULA, 
                                          window_size=3)

        return ff_model.summary(), results_df




    def beta(self):
        ff_data = self.data

        covariance = ff_data[['mkt','portf_ex_rtn']].cov().iloc[0,1]
        benchmark_variance = ff_data.mkt.var()
        return covariance / benchmark_variance # beta



    def automate_analysis(self): # FIXME
        statements = []

        beta = self.beta()

        model = self.ff_model
        coef_df = pd.DataFrame({'varname': model.params.index.values,
                                'coef': model.params.values,
                                # 'ci_err': model.params.ci.values,
                                'pvalue': model.pvalues.round(4).values})
        print(coef_df)
        intercept_p = coef_df.loc[coef_df.varname == 'Intercept'].coef.values[0]
        significant = coef_df.loc[coef_df.pvalue <= 0.05]

        statements.append(f'The alpha is {intercept_p}')
        statements.append(f'The beta is {beta}')
        for ix, row in significant.iterrows():
            statements.append(f"Based on A 1% change in the {row['varname']} factor, we can expect our portfolio to return {round(row['coef']*0.01,4)}% in excess of the risk free rate.")
        self.statements = statements



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
