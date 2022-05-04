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
    
    ticker = request.POST.get("tickers")
    if ticker in ['', None, 'None'] or len(str(ticker)) < 1:
        ticker = 'SPY'
    print(ticker)

    # TODO: call from retreiver
    import pandas_datareader as data
    import pandas as pd
    data_source = 'yahoo'
    start_date = '2000-01-01'
    end_date = Calendar().today()
    stockdata = data.DataReader(ticker, data_source, start_date, end_date)['Close']
    stockdata =  pd.DataFrame(stockdata).rename(columns={'Close':ticker})

    train_start = request.POST.get("train_start")
    train_end = request.POST.get("train_end")
    test_start = request.POST.get("test_start")
    test_end = request.POST.get("test_end")
    if train_start in ['', None, 'None']:
        train_start = -0.6
        train_end = -0.10
        test_start = -0.4
        test_end = -0.01

    # FIXME
    ts = TimeSeries(data=stockdata, column=ticker)
    ts.decomposition(model='additive')
    ts.check_stationarity()
    ts.auto_correlation( lags=24 )
    ts.prophet_forecast(int(train_start), int(train_end), int(test_start), int(test_end))

    img_dirp = os.path.join(os.getcwd(), 'pynance/static/img')

    print(f'{ticker}_forecast.png')
    context = {
        'seasonal_decompose.png':os.path.join(img_dirp, 'seasonal_decompose.png'),
        'actual_v_predicted.png':os.path.join(img_dirp, 'actual_v_predicted.png'),
        'prophet_forecast.png':os.path.join(img_dirp, 'prophet_forecast.png'),
        'autocorrelation.png':os.path.join(img_dirp, 'autocorrelation.png'),
        'components.png':os.path.join(img_dirp, 'components.png'),
    }

    return render(request, 'forecast.html', context)