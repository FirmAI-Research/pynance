from django.shortcuts import render

from django.templatetags.static import static
from pathlib import Path
import sys, os
import json
import yfinance as yf

proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(proj_root, 'lib'))
sys.path.append(os.path.join(proj_root, 'lib', 'fixed_income'))

import calendar_dates
cal = calendar_dates.Calendar()

from interest_rates import Treasuries
import inflation_rates
import credit_spreads

import wsj_bond_benchmarks
import pandas as pd

import webtools

fp_ajax = os.path.join( Path(__file__).resolve().parent, 'ajax')



def treasuries(request):

    ust = Treasuries(years = ['2022'])
    print(ust.df)

    recent_rates = ust.df.iloc[-10:]
    recent_rates_response = webtools.df_to_highcharts_heatmap(recent_rates)
    
    recent_rates_change = ust.df.iloc[-10:].diff(axis=0).dropna(axis=0).round(2)
    recent_rates_change_response = webtools.df_to_highcharts_heatmap(recent_rates_change)

    change_since = ust.change_since()
    change_since_response = webtools.df_to_highcharts_clustered_bar(change_since)

    data = yf.download(f"TLT AGG BND SPY", start="2015-01-01", end=cal.today())['Adj Close']
    stock_bond_corr = ust.market_correlations(data, title = None)
    stock_bond_corr_response = webtools.df_to_highcharts_heatmap(stock_bond_corr)

    stock_ust_corr = ust.stock_treasury_correlation()
    stock_ust_corr_response = webtools.df_to_highcharts_heatmap(stock_ust_corr)

    tens_twos = ust.tens_twos_spread()
    tens_twos_response = webtools.df_to_highcharts_linechart(tens_twos)

    context = {

        'recent_rates_response':recent_rates_response,
        'recent_rates_change_response':recent_rates_change_response,
        'change_since_response':change_since_response,
        'stock_bond_corr_response':stock_bond_corr_response,
        'stock_ust_corr_response':stock_ust_corr_response,
        'tens_twos_response':tens_twos_response

    }

    return render(request, 'treasuries.html', context)



def inflation(request):

    breakeven = inflation_rates.breakeven_inflation()
    breakeven_response = webtools.df_to_highcharts_linechart(breakeven)

    expected_inflation = inflation_rates.expected_inflation()
    expected_inflation_response = webtools.df_to_highcharts_linechart(expected_inflation)
    print(expected_inflation_response)





    context = {
        'breakeven_response':breakeven_response,
        'expected_inflation_response':expected_inflation_response
    }

    return render(request, 'inflation.html', context)



def fixed_income(request):

    bbm = wsj_bond_benchmarks.BondBenchmarks()
    bbm.get().parse()
    print(bbm.df)

    fp = os.path.join(fp_ajax, 'bond_benchmarks.json')
    benchmarks_response = webtools.df_to_dt(bbm.df.fillna('-').replace('n.a.', '-'), fp)


    credit_spread = credit_spreads.get_credit_spreads()
    credit_spread_response = webtools.df_to_highcharts_linechart(credit_spread)


    context = {
        'benchmarks_response':benchmarks_response,
        'credit_spread_response':credit_spread_response,
    }

    return render(request, 'fixed_income.html', context)