import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 



def autoregression(data=None, col_name=None,):
    #https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
    from pandas import read_csv
    from matplotlib import pyplot
    from statsmodels.tsa.ar_model import AutoReg
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    # load dataset
    series = data[col_name]
    # split dataset
    X = series.values
    train, test = X[1:len(X)-7], X[len(X)-7:]
    # train autoregression
    model = AutoReg(train, lags=29)
    model_fit = model.fit()
    print('Coefficients: %s' % model_fit.params)
    # make predictions
    predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot results
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()


def arima(data=None, col_name=None,):
    #https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    from pandas import read_csv
    from pandas import datetime
    from matplotlib import pyplot
    from pandas.plotting import autocorrelation_plot
    from statsmodels.tsa.arima.model import ARIMA

    def parser(x):
        return datetime.strptime('190'+x, '%Y-%m')
    
    series = data[col_name]
    autocorrelation_plot(series)
    plt.show()
    series.index = series.index.to_period('M')
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())

    #seasonal ARIMA; ARCH; GARCH
    