from django.shortcuts import render

from lib.fixed_income.rates.treasury_rates import TreasuryRates
from lib.learn.regression.linreg import _StatsModels
from lib.calendar import Calendar
cal = Calendar()

import pandas as pd
import numpy as np 
import yfinance as yf

def index(request):

    tr = TreasuryRates()
    rates_table = tr.get()   # FIXME: format for use w HighCharts.fromJSON()
    rates_table.date = rates_table.date.apply(lambda x : x.strftime('%Y-%m-%d'))

    points_in_time_json = tr.point_in_time_curves()

    all_tenors_over_time, all_tenors_x_axis = tr.all_tenors_over_time()
    
    weekly_change, change_x_axis = tr.change_distribution()

    weekly_change_spider, change_x_axis_spider = tr.change_distribution_spider()


    context = {
        
        'rates_table':rates_table,
        'values':rates_table.values.tolist(),

        'points_in_time_json':points_in_time_json,

        'all_tenors_over_time':all_tenors_over_time,
        'all_tenors_x_axis':all_tenors_x_axis,

        'weekly_change':weekly_change,
        'change_x_axis':change_x_axis,

        'weekly_change_spider':weekly_change_spider,
        'change_x_axis_spider':change_x_axis_spider

    }

    return render(request, 'treasury_rates.html', context)



def market_return_regression(request):


    ticker = str(request.POST.get("tickers"))
    tenors = request.POST.get("tenors")
    x_col_str = tenors

    try: 
        date_start = pd.to_datetime(str(request.POST.get("datepicker_start"))).strftime('%Y-%m-%d')
    except Exception:
        date_start = '2022-01-01'

    if ticker == '' or ticker == 'None' or ticker is None or len(str(ticker)) < 1:
        ticker = 'IVE'
    if x_col_str is None:
        x_col_str = '10 Year'
    print(ticker)
    print(date_start)

    x_str = x_col_str + ' Treasury'
    date_end = cal.today()

    tr = TreasuryRates()
    rates_table = tr.get().rename(columns = {'date':'Date'}).set_index('Date')   # FIXME: format for use w HighCharts.fromJSON()

    prices = pd.DataFrame(yf.download(ticker, date_start, date_end)['Adj Close'].pct_change().dropna())
    prices.rename(columns = {'Adj Close':ticker}, inplace = True)

    df = prices.merge(rates_table, left_index=True, right_index = True)
    print(df)

    sm = _StatsModels(df = df, x_col_str = x_col_str, y_col_str = ticker)
    print(sm.model.summary())

    model_summary = sm.model.summary().as_html().replace(
        '<table class="simpletable">', '').replace('</table>', '')

    context = { 
        'ticker':ticker,
        'x_str':x_str,
        'date_start': date_start,
        'date_end': date_end,
        'model_summary':model_summary,
    
    }

    return render(request, 'market_return_regression.html', context)