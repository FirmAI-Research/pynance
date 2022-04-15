import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

class Regression:

    def __init__(self):
        pass



    def evaluate_fit(self):
        ''' evaluate model fit after calling sm.OLS().fit()'''
        plt.scatter(self.x,self.y)
        plt.show()
        self.y_true = self.model.params[0] + self.model.params[1]*self.x
        plt.plot(self.x, self.y_true, linewidth=3)
        plt.xlabel(self.x_col_str)
        plt.ylabel(self.y_col_str)
        plt.title('OLS Regression')
        self.results = self.model.summary()
        print(self.results)
        return self.results


    def evaluate_predictions(self):
        ''' evaluate accuracy of predictions using fit model after making out of sample predictions '''
        fig, ax = plt.subplots()
        ax.plot(self.x, self.y, "o", label="Data")
        ax.plot(self.x, self.y_true, "b-", label="True")
        ax.plot(np.hstack((self.x, self.x1n)), np.hstack((self.ypred, self.ynewpred)), "r", label="OLS prediction")
        ax.legend(loc="best")



from sklearn.linear_model import LinearRegression

class _Sklearn(Regression):

    def __init__(self, df:pd.DataFrame, x_col_str, y_col_str):
        super().__init__()

        self.df = df
        self.x_col_str = x_col_str
        self.y_col_str = y_col_str
        
        self.x = self.df[self.x_col_str]
        self.y = self.df[self.y_col_str]
        self.model = LinearRegression().fit(self.x, self.y) 
        r_sq = self.model.score(self.x, self.y)

        print('coefficient of determination:', r_sq)
        print('intercept:', self.model.intercept_)
        print('slope:', self.model.coef_) 
        y_pred = self.model.predict(self.x)
        print('Predicted response:', y_pred, sep='\n')



import statsmodels.api as sm

class _StatsModels(Regression):

    def __init__(self, df:pd.DataFrame, x_col_str, y_col_str):
        super().__init__()

        self.df = df
        self.x_col_str = x_col_str
        self.y_col_str = y_col_str
        
        print('Using Simple Linear Regression') if isinstance(self.x_col_str, str) else print('Using Multiple Regression') if isinstance(self.x_col_str, list) else ValueError('x_col_str must be str or list of str')
        self.x = self.df[self.x_col_str]
        self.y = self.df[self.y_col_str]
        self.X = sm.add_constant(self.x)
        self.model = sm.OLS(self.y, self.X).fit()


    def predict_in_sample(self):
        self.ypred  = self.model.predict(self.X) 
        return self.ypred
    
    
    def predict_out_of_sample(self, n:int=10):
        self.x1n = np.linspace(-n, n, 1)
        self.Xnew = sm.add_constant(self.x1n)
        self.ynewpred = self.model.predict(self.Xnew)






        