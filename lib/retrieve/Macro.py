
class Macro(): # @ todo

    """
    retrieve macro economic data and broader market/indices data
    
    :param w: 
    :type w: 
    :param obj: 
    :type obj: 
    :return: 
    :rtype: df
    """

    def __init(self):
        pass

    def scrape_wiki(self):
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        df = pd.read_html(url, header=0)[0]
        df.columns = ['ticker', 'name', 'sec_filings', 'gics_sector', 'gics_sub_industry',
                    'location', 'first_added', 'cik', 'founded']
        df = df.drop('sec_filings', axis=1).set_index('ticker')
        print(df.head())
        #print(df.info())


# @ todo

#fred 

def get_fred(self):
    start = datetime(2010, 1, 1)

    end = datetime(2013, 1, 27)

    gdp = web.DataReader('GDP', 'fred', start, end)

    gdp.info()

    inflation = web.DataReader(['CPIAUCSL', 'CPILFESL'], 'fred', start, end)
    inflation.info()

def fama_french():
    from pandas_datareader.famafrench import get_available_datasets
    get_available_datasets()

    ds = web.DataReader('5_Industry_Portfolios', 'famafrench')
    print(ds['DESCR'])


def world_bank():
    from pandas_datareader import wb
    gdp_variables = wb.search('gdp.*capita.*const')
    gdp_variables.head()



def oecd():
    df = web.DataReader('TUD', 'oecd', end='2015')
    df[['Japan', 'United States']]

def euro_stat():
    df = web.DataReader('tran_sf_railac', 'eurostat')


import pandas as pd 

import pandas_datareader as pdr
# load gold data from FRED API & save copy locally to CSV file
series = ['GOLDAMGBD228NLBM']
gold_download = pdr.data.DataReader(series, 
                                    'fred', 
                                    start='1968-12-31')
# convert daily to annual
gold_download = gold_download.resample('A').last().reset_index()
gold_download.set_index(pd.DatetimeIndex(gold_download['DATE']).year, inplace=True)
gold_download['return'] = gold_download['GOLDAMGBD228NLBM'].pct_change()
gold_download.to_csv('gold_fred.csv')



series = ['GDPCA']

gdp_download = pdr.data.DataReader(series, 
                                   'fred', 
                                   start='1926-12-31')
gdp_download.reset_index(inplace=True)
gdp_download.set_index(pd.DatetimeIndex(gdp_download['DATE']).year, inplace=True)
gdp_download['GDP'] = gdp_download['GDPCA'].pct_change()
# https://fortunly.com/statistics/us-gdp-by-year-guide/#gref
gdp_download.loc[1928, 'GDP'] = 0.0110
gdp_download.loc[1929, 'GDP'] = 0.0652
gdp_download.to_csv('gdp_fred.csv')

print(gdp_download)



import pandas_datareader.data as web

securities = {'BAMLCC0A0CMTRIV'   : 'US Corp Master TRI',
              'BAMLHYH0A0HYM2TRIV': 'US High Yield TRI',
              'BAMLEMCBPITRIV'    : 'Emerging Markets Corporate Plus TRI',
              'GOLDAMGBD228NLBM'  : 'Gold (London, USD)',
              'DGS10'             : '10-Year Treasury CMR',
              }

df = web.DataReader(name=list(securities.keys()), data_source='fred', start=2000)
df = df.rename(columns=securities).dropna(how='all').resample('B').mean()
print(df)






data_xls = 'http://www.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls'
data_sheet = "Returns by year"
# these will change as rows get added on Damodaran website
skiprows = range(17)
skipfooter = 10
download_df = pd.read_excel('http://www.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls', 
                         sheet_name=data_sheet, 
                         skiprows=skiprows,
                         skipfooter=skipfooter)
download_df = download_df.set_index('Year')
data_df = download_df.copy()
data_df = data_df.rename(columns = {'Inflation Rate': 'CPI',
           'S&P 500 (includes dividends)2': 'S&P',
           '3-month T. Bill (Real)': 'T-Bills',
           '!0-year T.Bonds': 'T-Notes',
           'Baa Corp Bonds': 'Baa Corps',
          })[['CPI', 'S&P', 'T-Bills', 'T-Notes', 'Baa Corps']]
data_df["GDP"] = gdp_download['GDP']
data_df["Gold"] = longrun_data['gold'] - data_df['CPI']
# reorder
data_df = data_df[['GDP', 'CPI', 'S&P', 'T-Bills', 'T-Notes', 'Baa Corps', 'Gold']]


#https://github.com/druce/portfolio_optimization/blob/master/Portfolio%20optimization.ipynb