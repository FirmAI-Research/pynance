import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE


class Regression:
    """[Summary]
    Pass and unindexed dataframe as data

    """    

    def __init__(self, data:pd.DataFrame, dep_var:str) -> None:
        self.df = data
        self.dep_var = dep_var

        # Checking for null values
        print(self.df.info())
        # Checking for outliers
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



    def split(self, test_size:float) -> None:
        self.df_train, self.df_test = train_test_split(self.df, train_size = 1-test_size, test_size = test_size) # random_state = 100; ensures train and test keep the same rows


    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns


    def scale(self):
        self.scaler = MinMaxScaler()
        numeric_cols = self.get_numeric_cols()
        self.df_train[numeric_cols] = self.scaler.fit_transform(self.df_train[numeric_cols])

    
    def build_model(self):
        self.y_train = self.df_train.pop(self.dep_var)
        self.X_train = self.df_train
        self.X_train_lm = sm.add_constant(self.X_train)
        self.model = sm.OLS(self.y_train, self.X_train_lm).fit()
        self.model_summary = self.model.summary()
        

    def vif(self):
        """[Summary]
        Variance Inflation Factor or VIF is a quantitative value that says how much the feature variables are correlated with each other. 
        Keep varibles with VIF values < 5
        If VIF > 5 and high p-value, drop the variable; Rinse and repeat until all variables have VIF < 5 and significant p-values (<0.005)
        """        
        self.vif = pd.DataFrame()
        self.vif['Features'] = self.X_train.columns
        self.vif['VIF'] = [variance_inflation_factor(self.X_train.values, i) for i in range(self.X_train.shape[1])]
        self.vif['VIF'] = round(self.vif['VIF'], 2)
        self.vif = self.vif.sort_values(by = "VIF", ascending = False)
        self.vif


    def check_residuals(self):
        """[Summary]
        check if the error terms are normally distributed 
        """
        y_train_price = self.model.predict(self.X_train_lm)
        # Plot the histogram of the error terms
        fig = plt.figure()
        sns.distplot((self.y_train - y_train_price), bins = 20)
        fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
        plt.xlabel('Errors', fontsize = 18)                         # X-label
        plt.show()

    
    def check_model(self):
        self.check_residuals()
        self.df_test[self.get_numeric_cols()] = self.scaler.transform(self.df_test[self.get_numeric_cols()])
        self.y_test = self.df_test.pop(self.dep_var)
        self.X_test = self.df_test
        self.X_test = sm.add_constant(self.X_test)
        self.y_pred = self.model.predict(self.X_test)
        print(r2_score(y_true = self.y_test, y_pred = self.y_pred))



    # def evaluate_predictions(self):
    #     ''' evaluate accuracy of predictions using fit model after making out of sample predictions '''
    #     fig, ax = plt.subplots()
    #     ax.plot(self.x, self.y, "o", label="Data")
    #     ax.plot(self.x, self.y_true, "b-", label="True")
    #     ax.plot(np.hstack((self.x, self.x1n)), np.hstack((self.ypred, self.ynewpred)), "r", label="OLS prediction")
    #     ax.legend(loc="best")




    def evaluate_fit(self):
        ''' evaluate model fit after calling sm.OLS().fit()'''
        plt.scatter(self.X_test,self.y_test)
        plt.show()
        # y_true = self.model.params[0] + self.model.params[1]*self.X_test
        # plt.plot(self.X_test, self.y_test, linewidth=3)
        # # plt.xlabel(self.x_col_str)
        # # plt.ylabel(self.y_col_str)
        # plt.title('OLS Regression')
        # self.results = self.model.summary()
        # print(self.results)
        return self.results


    # def rfe(self):
    #     # Running RFE with the output number of the variable equal to 10
    #     lm = LinearRegression()
    #     lm.fit(X_train, y_train)
    #     rfe = RFE(lm, 10)             # running RFE
    #     rfe = rfe.fit(X_train, y_train)
    #     list(zip(X_train.columns,rfe.support_,rfe.ranking_))



# Test
# from lib.learn.regression.regression import Regression
# fp = r"C:\dev\pynance\_tmp\housing.csv"
# df = pd.read_csv(fp)
# reg = Regression(data=df, dep_var='price')
# reg.binary_encode(var_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])
# reg.cat_encode(var_list = ['furnishingstatus'])
# reg.split(test_size=0.3)
# reg.scale()
# reg.build_model()
# print(reg.df)
# print(reg.model_summary)
# reg.vif()
# print(reg.vif)
# reg.check_model()
