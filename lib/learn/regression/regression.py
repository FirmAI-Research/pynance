import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std


class Regression:
    """[Summary]
    Pass and unindexed dataframe as data

    """    
    def __init__(self, data:pd.DataFrame, dep_var:str, indep_var:str=None) -> None:
        self.df = data
        self.dep_var = dep_var

        if indep_var != None:
            self.how = 'univariate'
            self.indep_var = indep_var
        else:
            self.how = 'multivariate'

        print(self.df.info())
        print(self.df.describe())


    def binary_encode(self, var_list:str) -> None:
        def binary_map(x):
            return x.map({'yes': 1, "no": 0})
        self.df[var_list] = self.df[var_list].apply(binary_map)


    def cat_encode(self, var_list:str) -> None:
        for var in var_list:
            dummies = pd.get_dummies(self.df[var])
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(var, axis=1, inplace=True)


    def split(self, test_size:float=0.4) -> None:
        self.df_train, self.df_test = train_test_split(self.df, train_size = 1-test_size, test_size = test_size) # random_state = 100; ensures train and test keep the same rows


    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns


    def cast_numeric_cols(self):
        for c in self.df.columns:
            if c in self.get_numeric_cols():
                self.df[c] = pd.to_numeric(self.df[c])


    def scale(self):
        self.scaler = MinMaxScaler()

    
    def train_model(self):
        df_train = self.df_train
        y_train = df_train[self.dep_var]
        df_train.pop(self.dep_var)
        print(df_train)
        # self.df_train[self.get_numeric_cols()] = self.scaler.fit_transform(df_train[self.get_numeric_cols()]) # FIXME
        X_train = df_train
        if self.how == 'univariate':
            X_train_lm = sm.add_constant(X_train[self.indep_var]) # Simple Linreg
        else:
            X_train_lm = sm.add_constant(X_train)  # Multiple regression
        self.model = sm.OLS(y_train.astype(float), X_train_lm.astype(float)).fit()
        model_summary = self.model.summary()
        print(model_summary)
    

    def reg_plots(self):
        if self.how == 'univariate':

            fig = plt.figure(figsize=(15,8))
            fig = sm.graphics.plot_regress_exog(self.model, self.indep_var, fig=fig)
            plt.show()

            def confidence_intervals():
                x = self.df_train[self.indep_var]
                y = self.df_train[self.dep_var]
                _, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(self.model)
                fig, ax = plt.subplots(figsize=(10,7))
                ax.plot(x, y, 'o', label="data")
                ax.plot(x, self.model.fittedvalues, 'g--.', label="OLS")
                ax.plot(x, confidence_interval_upper, 'r--')
                ax.plot(x, confidence_interval_lower, 'r--')
                ax.legend(loc='best')
                plt.show()
            confidence_intervals()
        
        elif self.how == 'multivariate':

            fig = plt.figure(figsize=(20,12))
            fig = sm.graphics.plot_partregress_grid(self.model, fig=fig)
            plt.show()


            
    def test_model(self):
        df_test = self.df_test
        y_test = df_test[self.dep_var]
        df_test.pop(self.dep_var)
        # df_test[self.get_numeric_cols()] = self.scaler.transform(df_test[self.get_numeric_cols()])    # FIXME
        X_test = df_test
        X_test_lm = sm.add_constant(X_test)
        y_pred = self.model.predict(X_test_lm)
        print(r2_score(y_true = y_test, y_pred = y_pred))

        def residuals_distribution():
            fig = plt.figure()
            sns.distplot((y_test - y_pred), )
            fig.suptitle('Error Terms', fontsize = 20)                   
            plt.xlabel('Errors', fontsize = 18)                         
            plt.show()
        residuals_distribution()

        def scatter():
            fig = plt.figure()
            sns.regplot(y_test, y_pred)
            fig.suptitle('Acctual minus Predicted', fontsize = 20)                   
            plt.xlabel('Actual', fontsize = 18)
            plt.ylabel('Predicted', fontsize = 18)
            plt.show()
        scatter()


    def oos_predict(self, X_new:pd.Series):
        ''' X_new is a series in the same shape of the X_train data
        ''' 
        X_new = pd.DataFrame(X_new).transpose() 
        # X_new[self.get_numeric_cols()] = self.scaler.transform(X_new[self.get_numeric_cols()]) 
        X_test_lm_new = sm.add_constant(X_new)
        y_pred_new = self.model.predict(X_test_lm_new)
        # df = pd.DataFrame(self.scaler.inverse_transform(X_new))
        X_new.columns = ['const'] + [c for c in self.df_train.columns if c != self.dep_var]
        print(X_new)
        print(y_pred_new)


    def oos_iterative_predict(self, n:int = 10):
        ''' Recursively predicts n new periods of test data, retraining the model on all test data as well as the newly predicted data after each iteration.
        n = number of forward periods to predict
        '''
        pass






    # def vif(self):
        """[Summary]
        Variance Inflation Factor or VIF is a quantitative value that says how much the feature variables are correlated with each other. 
        Keep varibles with VIF values < 5
        If VIF > 5 and high p-value, drop the variable; Rinse and repeat until all variables have VIF < 5 and significant p-values (<0.005)
        """        
        # vif = pd.DataFrame()
        # vif['Features'] = self.X_train.columns
        # vif['VIF'] = [variance_inflation_factor(self.X_train.values, i) for i in range(self.X_train.shape[1])]
        # vif['VIF'] = round(vif['VIF'], 2)
        # vif = vif.sort_values(by = "VIF", ascending = False).to_dict()
        # vif
        # pass


    # def evaluate_predictions(self):
    #     ''' evaluate accuracy of predictions using fit model after making out of sample predictions '''
    #     fig, ax = plt.subplots()
    #     ax.plot(self.x, self.y, "o", label="Data")
    #     ax.plot(self.x, self.y_true, "b-", label="True")
    #     ax.plot(np.hstack((self.x, self.x1n)), np.hstack((self.ypred, self.ynewpred)), "r", label="OLS prediction")
    #     ax.legend(loc="best")


    # def rfe(self):
    #     # Running RFE with the output number of the variable equal to 10
    #     lm = LinearRegression()
    #     lm.fit(X_train, y_train)
    #     rfe = RFE(lm, 10)             # running RFE
    #     rfe = rfe.fit(X_train, y_train)
    #     list(zip(X_train.columns,rfe.support_,rfe.ranking_))
