# evaluate an ARIMA model using a walk-forward validation
import warnings
from math import sqrt
import pandas as pd
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA as _ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
warnings.filterwarnings("ignore")
from matplotlib import pyplot


from timeseries import TimeSeries 

class Arima(TimeSeries):
    '''
    Auto Regressive Integrated Moving Average
    '''

    def __int__(self, fp:str=None):
        super(self).__init__(fp)
        print(self.data)


    def Arima(self):
        self.data.index = pd.to_datetime(self.data.index)
        self.series = self.data['Close']
        self.series.index = self.series.index.to_period('M')
        # fit model
        model = _ARIMA(self.series, order=(5,1,0))
        model_fit = model.fit()
        # summary of fit model
        print(model_fit.summary())
        # line plot of residuals
        residuals = pd.DataFrame(model_fit.resid)
        residuals.plot()
        pyplot.show()
        # density plot of residuals
        residuals.plot(kind='kde')
        pyplot.show()
        # summary stats of residuals
        print(residuals.describe())
        return self


    '''   walk-forward validation    '''
    
    def evaluate(self):
        X = self.series.values
        size = int(len(X) * 0.001)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()

        for t in range(len(test)):
            model = _ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            print('predicted=%f, expected=%f' % (yhat, obs))
        # evaluate forecasts
        rmse = sqrt(mean_squared_error(test, predictions))
        print('Test RMSE: %.3f' % rmse)
        # plot forecasts against actual outcomes
        pyplot.plot(test, linewidth=1, linestyle=':')
        pyplot.plot(predictions, color='red', linewidth=1, linestyle='-')
        # pyplot.ylim(30, 150)
        pyplot.show()



    ''' hyper-parameter tuning '''
    # evaluate an ARIMA model for a given order (p,d,q)
    def evaluate_arima_model(self, X, arima_order):
        # prepare training dataset
        train_size = int(len(X) * 0.66)
        train, test = X[0:train_size].reset_index(drop=True), X[train_size:].reset_index(drop=True)
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = _ARIMA(history, order=arima_order)
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse

    # evaluate combinations of p, d and q values for an ARIMA model
    def evaluate_models(self, p_values, d_values, q_values):
        dataset = self.data.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        rmse = self.evaluate_arima_model(dataset, order)
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order
                        print('ARIMA%s RMSE=%.3f' % (order,rmse))
                    except:
                        continue
        print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))