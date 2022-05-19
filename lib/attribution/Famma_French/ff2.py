 # famma french from pakt cookbook


def calc_betas():
    bm = calc_bm
    subadvisor_strs = [x for x in merge.columns[3:]]
    betas = []
    for subadv in subadvisor_strs:
        covariance = merge[[subadv,calc_bm]].cov().iloc[0,1]
        benchmark_variance = merge[[calc_bm]].var()
        beta = covariance / benchmark_variance
        betas.append(beta)
    x = pd.DataFrame(pd.concat(betas)).transpose()
    x.columns =  subadvisor_strs
    return x



def capm():
    _merge = merge.copy()
    for subadv in subadvisor_strs:

        # separate target
        y = _merge.pop(subadv)

        # add constant
        X = sm.add_constant(_merge[calc_bm])

        # define and fit the regression model 
        capm_model = sm.OLS(y, X).fit()

        
        y = merge[subadv]                 # dependant variable
        x = merge[calc_bm]                 # independent variables  
        # print results 
        print(capm_model.summary())
        # results = model.fit()
        plt.figure(figsize=(20,5))
        plt.scatter(y,y,alpha=0.3)
        y_true = y_predict = capm_model.params[0] + capm_model.params[1]*x
        plt.plot(x, y_predict, linewidth=3)
        plt.xlabel(calc_bm)
        plt.ylabel(subadv)
        plt.legend()
        plt.title('OLS Regression')
        plt.show()
        
def sensitivity_analysis():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    frames = []
    for y_str in [c for c in returns.columns if c != 'Date']:

        y = merge[y_str]                      
        x = merge[calc_bm]                  

        X = sm.add_constant(x)
        model = sm.OLS(y,X).fit()

        x1n = np.linspace(-5, 5, 10)
        Xnew = sm.add_constant(x1n)
        ynewpred = model.predict(Xnew)
        
        res = pd.DataFrame([x1n, ynewpred]).transpose().rename(columns={0:calc_bm, 1:y_str})
        res.set_index(calc_bm, inplace=True)
        frames.append(res)

    import pandas as pd
    from functools import reduce
    df = reduce(lambda x, y: pd.merge(x, y, on = calc_bm), frames)
    df


def rolling_factor_model(input_data, formula, window_size):
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


def three_factor():
    START_DATE = '2014-01-01'
    ff_dict = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                            start=START_DATE)    
    print(ff_dict['DESCR'])        
    df_three_factor = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 
                                    start=START_DATE)[0]
    df_three_factor = df_three_factor.div(100)
    df_three_factor.index = df_three_factor.index.strftime('%Y-%m-%d')
    df_three_factor.index.name = 'Date'            
    asset_df = merge
    asset_df.set_index('Date', inplace = True)    
    asset_df.drop('S&P 500 PR', axis=1,inplace = True)
    asset_df.tail() 


    plt.rcParams["figure.figsize"] = (20,7)


    for subadv in subadvisor_strs:

        ff_data = asset_df[[subadv]].join(df_three_factor)
        ff_data.columns = ['portf_rtn', 'mkt', 'smb', 'hml', 'rf']
        ff_data['portf_ex_rtn'] = ff_data.portf_rtn - ff_data.rf
        
        ff_model = smf.ols(formula='portf_ex_rtn ~ mkt + smb + hml',
        data=ff_data).fit()
        print(f'Results for: {subadv}')
        print(ff_model.summary())

        for c in ff_data.columns:
            ff_data[c] = pd.to_numeric(ff_data[c])
        MODEL_FORMULA = 'portf_ex_rtn ~ mkt + smb + hml'
        results_df = rolling_factor_model(ff_data, 
                                        MODEL_FORMULA, 
                                        window_size=12)
        print(results_df.tail())    
        results_df.plot(title = f'{subadv} - Rolling Fama-French Three-Factor model')
        plt.show()
    