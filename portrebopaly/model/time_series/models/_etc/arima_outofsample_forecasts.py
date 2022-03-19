#https://machinelearningmastery.com/make-sample-forecasts-arima-python/
# line plot of time series
import sys, os
iop = os.getcwd() +'/io/'

from pandas import read_csv
from matplotlib import pyplot
# load dataset
series_name = 'X'

import yfinance as yf
series = yf.download(series_name,'2015-01-01')['Adj Close'].squeeze().dropna()# display first few rows
print(series.head(20))
# line plot of dataset
series.plot()
pyplot.show()


split_point = len(series) - 7
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv(f'{iop}dataset.csv', index=False)
validation.to_csv(f'{iop}validation.csv', index=False)


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return numpy.array(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


from statsmodels.tsa.arima.model import ARIMA
import numpy



# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit()
# print summary of fit model
print(model_fit.summary())






from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy

# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)




# one-step out of sample forecast
# start_index = len(differenced)
# end_index = len(differenced)
# forecast = model_fit.predict(start=start_index, end=end_index)
# start_index = '2019-12-25'
# end_index = '2019-12-25'
# forecast = model_fit.predict(start=start_index, end=end_index)

# from pandas import datetime
# start_index = datetime(1990, 12, 25)
# end_index = datetime(1990, 12, 26)
# forecast = model_fit.predict(start=start_index, end=end_index)

# # one-step out of sample forecast
# start_index = len(differenced)
# end_index = len(differenced)
# forecast = model_fit.predict(start=start_index, end=end_index)
# # invert the differenced forecast to something usable
# forecast = inverse_difference(X, forecast, days_in_year)
# print('Forecast: %f' % forecast)







#predict
# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import numpy

# seasonal difference
X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)
# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit()
# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1