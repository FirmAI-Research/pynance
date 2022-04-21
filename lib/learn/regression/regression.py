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

        print(self.df.info())
        print(self.df.describe())

    # @ TODO move to feature engine
    def binary_encode(self, var_list:str) -> None:
        def binary_map(x):
            return x.map({'yes': 1, "no": 0})
        self.df[var_list] = self.df[var_list].apply(binary_map)

    # @ TODO move to feature engine
    def cat_encode(self, var_list:str) -> None:
        for var in var_list:
            dummies = pd.get_dummies(self.df[var])
            self.df = pd.concat([self.df, dummies], axis=1)
            self.df.drop(var, axis=1, inplace=True)


    def split(self, test_size:float) -> None:
        self.df_train, self.df_test = train_test_split(self.df, train_size = 1-test_size, test_size = test_size) # random_state = 100; ensures train and test keep the same rows


    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns


    def cast_numeric_cols(self):
        for c in self.df.columns:
            if c in self.get_numeric_cols():
                self.df[c] = pd.to_numeric(self.df[c])


    def scale(self):
        self.scaler = MinMaxScaler()
        print('scale column order: ', self.df.columns)
        # numeric_cols = self.get_numeric_cols()

    
    def train_model(self):
        df_train = self.df_train
        df_train[self.get_numeric_cols()] = self.scaler.fit_transform(df_train[self.get_numeric_cols()])
        y_train = df_train.pop(self.dep_var)
        X_train = df_train
        X_train_lm = sm.add_constant(X_train)
        self.model = sm.OLS(y_train.astype(float), X_train_lm.astype(float)).fit()
        self.model_summary = self.model.summary()
        
    
    def test_model(self):
        df_test = self.df_test
        df_test[self.get_numeric_cols()] = self.scaler.transform(df_test[self.get_numeric_cols()])
        # X_actual = df_test[self.dep_var]
        y_test = df_test.pop(self.dep_var) # save as variable
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
        X_new = pd.DataFrame(X_new).transpose() # X_new is a series in the same shape of the X_train data
        X_new.fillna(0, inplace=True) # fill na value from most recent shift row as the most recent close price?
        X_new[self.get_numeric_cols()] = self.scaler.transform(X_new[self.get_numeric_cols()]) 
        X_test_lm_new = sm.add_constant(X_new)
        y_pred_new = self.model.predict(X_test_lm_new)
        X_new[self.dep_var] = y_pred_new
        df = pd.DataFrame(self.scaler.inverse_transform(X_new))
        df.columns = X_new.columns
        print(df)


    def oos_iterative_predict(self):
        pass


    def vif(self):
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
        pass


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
