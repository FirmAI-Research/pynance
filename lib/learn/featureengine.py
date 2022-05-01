import pandas as pd
import numpy as np
from dateutil.parser import parse
import regex as re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


class FeatureEngine:
    def __init__(self, df):
        self.df = df

        from sklearn.model_selection import train_test_split
        self.X, self.y = self.df.iloc[:, 1:].values, self.df.iloc[:, 0].values
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.3, 
                            stratify=self.y,
                            random_state=0)

        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        self.X_train_std = sc.fit_transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)


    def eig(self):
        '''numpy.linalg.eig function to decompose the symmetric covariance matrix into its eigenvalues and eigenvectors.
        '''
        cov_mat = np.cov(self.X_train_std.T)
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        print('\nEigenvalues \n%s' % eigen_vals)
        return eigen_vals, eigen_vecs

    def explained_variance(self):
        eigen_vals = self.eig()[0]
        tot = sum(eigen_vals)
        var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        print('\Explained Variance \n%s' % cum_var_exp)

        import matplotlib.pyplot as plt
        plt.bar(range(1, 14), var_exp, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(1, 14), cum_var_exp, where='mid',
                label='Cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal component index')
        plt.legend(loc='best')
        plt.tight_layout()
        # plt.savefig('images/05_02.png', dpi=300)
        plt.show()



    def get_numeric_cols(self):
        return self.df.select_dtypes(exclude=['object', 'datetime64']).columns


    def cast_numeric_cols(self):
        for c in self.df.columns:
            if c in self.get_numeric_cols():
                self.df[c] = pd.to_numeric(self.df[c])


    def get_cat_feats(data=None):
        '''
        Returns the categorical features in a data set
        '''
        if data is None:
            raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

        cat_features = data.select_dtypes(include=['object']).columns

        return list(cat_features)


    def get_num_feats(data=None):
        '''
        Returns the numerical features in a data set
        '''
        if data is None:
            raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

        num_features = data.select_dtypes(exclude=['object', 'datetime64']).columns

        return list(num_features)

                    
    def convert_dtype(df):
        '''
        Convert datatype of a feature to its original datatype.
        If the datatype of a feature is being represented as a string while the initial datatype is an integer or a float 
        or even a datetime dtype. The convert_dtype() function iterates over the feature(s) in a pandas dataframe and convert the features to their appropriate datatype
        '''
        if df.isnull().any().any() == True:
            raise ValueError("DataFrame contain missing values")
        else:
            i = 0
            changed_dtype = []
            #Function to handle datetime dtype
            def is_date(string, fuzzy=False):
                try:
                    parse(string, fuzzy=fuzzy)
                    return True
                except ValueError:
                    return False
                
            while i <= (df.shape[1])-1:
                val = df.iloc[:,i]
                if str(val.dtypes) =='object':
                    val = val.apply(lambda x: re.sub(r"^\s+|\s+$", "",x, flags=re.UNICODE)) #Remove spaces between strings
            
                try:
                    if str(val.dtypes) =='object':
                        if val.min().isdigit() == True: #Check if the string is an integer dtype
                            int_v = val.astype(int)
                            changed_dtype.append(int_v)
                        elif val.min().replace('.', '', 1).isdigit() == True: #Check if the string is a float type
                            float_v = val.astype(float)
                            changed_dtype.append(float_v)
                        elif is_date(val.min(),fuzzy=False) == True: #Check if the string is a datetime dtype
                            dtime = pd.to_datetime(val)
                            changed_dtype.append(dtime)
                        else:
                            changed_dtype.append(val) #This indicate the dtype is a string
                    else:
                        changed_dtype.append(val) #This could count for symbols in a feature
                
                except ValueError:
                    raise ValueError("DataFrame columns contain one or more DataType")
                except:
                    raise Exception()

                i = i+1

            data_f = pd.concat(changed_dtype,1)

            return data_f


    def display_missing(data=None, plot=False):
        '''
        Display missing values as a pandas dataframe.
        '''
        if data is None:
            raise ValueError("data: Expecting a DataFrame or Series, got 'None'")

        df = data.isna().sum()
        df = df.reset_index()
        df.columns = ['features', 'missing_counts']

        missing_percent = round((df['missing_counts'] / data.shape[0]) * 100, 1)
        df['missing_percent'] = missing_percent

        if plot:
            sns.heatmap(data.isnull(), cbar=True)
            plt.show()
            return df
        else:
            return df


    def detect_outliers(data, n, features):
        '''
            Detect Rows with outliers.
            Parameters
            '''
        outlier_indices = []

        # iterate over features(columns)
        for col in features:
            # 1st quartile (25%)
            Q1 = np.percentile(data[col], 25)
            # 3rd quartile (75%)
            Q3 = np.percentile(data[col], 75)
            # Interquartile range (IQR)
            IQR = Q3 - Q1

            # outlier step
            outlier_step = 1.5 * IQR

            # Determine a list of indices of outliers for feature col
            outlier_list_col = data[(data[col] < Q1 - outlier_step) | (data[col] > Q3 + outlier_step)].index

            # append the found outlier indices for col to the list of outlier indices
            outlier_indices.extend(outlier_list_col)

        # select observations containing more than 2 outliers
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        return multiple_outliers



    def facet_grid_example():
        sector_ts = xfer.copy()[['XFER_DATE','ASSET_CLASS_LONG_NAME','NET','CREDITS','DEBITS']].groupby(
            ['XFER_DATE','ASSET_CLASS_LONG_NAME']).sum()
        sector_ts['Rolling_Average_Long'] =  sector_ts[indicator].rolling(long_periods).mean()
        sector_ts['Rolling_Average_Short'] =  sector_ts[indicator].rolling(short_periods).mean()
        sector_ts.reset_index(inplace=True)
        sector_ts.dropna(inplace=True)
        sector_ts.drop(columns = [x for x in ['NET','CREDITS','DEBITS'] if x != indicator], inplace=True)
        sector_ts = sector_ts.loc[pd.to_datetime(sector_ts['XFER_DATE']) >= pd.to_datetime(ytd)]
        sector_ts = sector_ts.melt(id_vars=['ASSET_CLASS_LONG_NAME','XFER_DATE'], value_vars=[indicator,'Rolling_Average_Long','Rolling_Average_Short'], 
                                var_name='period', value_name='value')
        # sector_ts
        d = {'color': ['C0', 'k', 'r'], "ls" : ["-","--",":"]}
        sns.set(font_scale = 1.3)
        g = sns.FacetGrid(sector_ts, col='ASSET_CLASS_LONG_NAME', hue='period', height = 12, col_wrap=2, 
                        sharex=False, sharey=False, despine=True, margin_titles=True, hue_kws=d)
        g.map(sns.lineplot, 'XFER_DATE', 'value',legend='full',markers=True)
        for ax in g.axes.flat:
            ax.yaxis.grid(True)
            ax.xaxis.grid(True)
            ax.legend()
        g.add_legend() #loc='upper center'
        g.set_xticklabels(rotation=0, horizontalalignment='center')


    def reg_plots_example():
        sns.set(font_scale=2) 
        # g = sns.scatterplot(x='Date', y='Value', hue = 'Manager', data = melt, s=200)
        nyear_melt = melt.loc[(pd.to_datetime(melt.Date).dt.date >= nyearago) & (pd.to_datetime(melt.Date).dt.date < today)]

        nyear_melt.Date = dates.datestr2num([x.strftime('%Y-%m-%d') for x in nyear_melt.Date])# lmplot needs int x axis
        g = sns.lmplot(x="Date", y="Value", hue="Manager", col="Manager", data=nyear_melt, height=10, aspect=.6)
        for ax in g.axes.flat:
            ax.tick_params(labelrotation=45)
            ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d"))