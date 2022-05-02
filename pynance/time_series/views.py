from django.shortcuts import render

# from lib.time_series.arima import Arima
from lib.time_series.timeseries import TimeSeries

from lib.calendar import Calendar
cal = Calendar()
import sys, os

def time_series_models(request):
    context = {

    }
    return render(request, 'time_series.html', context)



def forecast(request):  
    
    # FIXME
    ts = TimeSeries(data='/Users/michaelsands/data/stock_prices.csv', column='AMZN')
    ts.decomposition(model='additive')
    ts.check_stationarity()
    ts.auto_correlation( lags=24 )
    ts.prophet_forecast()


    img_dirp = os.path.join(os.getcwd(), 'pynance/static/img')

    context = {
        'seasonal_decompose.png':os.path.join(img_dirp, 'seasonal_decompose.png'),
        'actual_v_predicted.png':os.path.join(img_dirp, 'actual_v_predicted.png'),
        'AMZN_forecast.png':os.path.join(img_dirp, 'AMZN_forecast.png'),
        'autocorrelation.png':os.path.join(img_dirp, 'autocorrelation.png'),
        'components.png':os.path.join(img_dirp, 'components.png'),
    }

    return render(request, 'forecast.html', context)