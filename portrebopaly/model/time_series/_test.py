
import pandas as pd
import numpy as np 

from arima import Arima
from timeseries import TimeSeries 


'''
Data sets for testing
'''
fp_stock = r"C:\Users\sands\Documents\_data\JPM.csv"
fp_factors = r"C:\Users\sands\Documents\_data\F-F_Research_Data_5_Factors_2x3.csv"
series = pd.read_csv(fp_stock)['Close'].dropna().iloc[:500]
df = pd.read_csv(fp_stock).dropna().iloc[:500]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace = True)


'''  
Tests for ARIMA Model
'''
#Arima(fp= fp_stock).Arima().evaluate()
#Arima(series = series).evaluate_models(p_values=[0, 1, 2, 4, 6, 8, 10], d_values=range(0, 3), q_values=range(0, 3))


'''
Tests for TimeSeries utility functions of class TimeSeries
'''
ts = TimeSeries(df=df)
# ts.interpolate_na(col_name = 'Close') 
# ts.decomposition(data = df, col_name ='Close', plot = True)
# ts.check_stationarity()
# ts.check_stationarity(col_name = 'Close')
# ts.auto_correlation(col_name = 'Close')