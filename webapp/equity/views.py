from django.shortcuts import render

from django.templatetags.static import static
from pathlib import Path
import sys, os
import json
import numpy as np 
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(proj_root, 'lib'))
sys.path.append(os.path.join(proj_root, 'lib', 'equity'))

from fundamentals import Fundamentals, Ranks, DCF, Columns
import webtools

fp_ajax = os.path.join( Path(__file__).resolve().parent, 'ajax')

from sqlalchemy import create_engine

import pandas as pd
import webtools


def fundamentals(request):

    print(request.POST)

    ticker = str(request.POST.get("inputTicker"))

    if ticker in ['', 'None', None]:
        ticker = 'BKNG'
    
    print(ticker)

    fun = Fundamentals( ticker = ticker)

    """ 
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │     Income Statement                                                                                                │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    inc = fun.get( columns = Columns.INCOME.value, limit = 5 ).for_js()
    fp = os.path.join(fp_ajax, 'income_statement.json')
    inc_json = webtools.df_to_dt(inc, fp)

    fun.percent_change()
    inc_pct_json = fun.pct_chg.T.multiply(100).reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
    inc_pct_json.columns.name = None
    fp = os.path.join(fp_ajax, 'income_statement_pct.json')
    inc_pct_json = webtools.df_to_dt(inc_pct_json.fillna('-'), fp)

    fun.quarter_over_quarter_change()
    inc_qq_change = fun.qq_change[fun.qq_change['index'].isin(Columns.INCOME.value)]
    inc_qq_change.replace([np.inf, -np.inf], np.nan, inplace=True)
    fp = os.path.join(fp_ajax, 'income_qq_change.json')
    inc_qq_json = webtools.df_to_dt(inc_qq_change.fillna('-'), fp)

    rank = Ranks(ticker = ticker)
    inc_ranks = rank.get_ranks().for_js(cols=Columns.INCOME_RANKS.value)
    fp = os.path.join(fp_ajax, 'income_statement_ranks.json')
    inc_ranks_json = webtools.df_to_dt(inc_ranks, fp)

    peers = fun.get_peers()
    peer_fun = Fundamentals(ticker = peers)
    inc_peer_values =  peer_fun.get( columns = Columns.INCOME.value, limit = 5 ).for_js(multiIndex = True)
    inc_peer_values.replace([np.inf, -np.inf], np.nan, inplace=True)
    fp = os.path.join(fp_ajax, 'income_statement_peer_values.json')    
    inc_peer_json =  webtools.df_to_dt(inc_peer_values.fillna('-'), fp)


    """ 
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Balance Sheet                                                                                                    │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    bs = fun.get( columns = Columns.BALANCE.value, limit = 5 ).for_js()
    fp = os.path.join(fp_ajax, 'balance_sheet.json')
    bs_json = webtools.df_to_dt(bs, fp)

    fun.percent_change()
    bs_pct_json = fun.pct_chg.T.multiply(100).reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
    bs_pct_json.columns.name = None
    fp = os.path.join(fp_ajax, 'balance_sheet_pct.json')
    bs_pct_json = webtools.df_to_dt(bs_pct_json.fillna('-'), fp)

    rank = Ranks(ticker = ticker)
    bs_ranks = rank.get_ranks().for_js(cols=Columns.BALANCE_RANKS.value)
    fp = os.path.join(fp_ajax, 'balance_sheet_ranks.json')
    bs_ranks_json = webtools.df_to_dt(bs_ranks, fp)

    peers = fun.get_peers()
    peer_fun = Fundamentals(ticker = peers)
    bs_peer_values =  peer_fun.get( columns = Columns.BALANCE.value, limit = 5 ).for_js(multiIndex = True)
    fp = os.path.join(fp_ajax, 'balance_sheet_peer_values.json')    
    bs_peer_json =  webtools.df_to_dt(bs_peer_values.fillna('-'), fp)


    """ 
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Cash Flow                                                                                                        │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    cf = fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).for_js()
    fp = os.path.join(fp_ajax, 'cash_flow.json')
    cf_json = webtools.df_to_dt(cf, fp)

    fun.percent_change()
    cf_pct_json = fun.pct_chg.T.multiply(100).reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
    cf_pct_json.columns.name = None
    fp = os.path.join(fp_ajax, 'cash_flow_pct.json')
    cf_pct_json = webtools.df_to_dt(cf_pct_json.fillna('-'), fp)

    rank = Ranks(ticker = ticker)
    cf_ranks = rank.get_ranks().for_js(cols=Columns.CASHFLOW_RANKS.value)
    fp = os.path.join(fp_ajax, 'cash_flow_ranks.json')
    cf_ranks_json = webtools.df_to_dt(cf_ranks, fp)

    peers = fun.get_peers()
    peer_fun = Fundamentals(ticker = peers)
    cf_peer_values =  peer_fun.get( columns = Columns.CASHFLOW_RANKS.value, limit = 5 ).for_js(multiIndex = True)
    fp = os.path.join(fp_ajax, 'cash_flow_peer_values.json')    
    cf_peer_json =  webtools.df_to_dt(cf_peer_values.fillna('-'), fp)


    industry = fun.industry

    context = {
        'ticker':ticker,
        'industry':industry,

        'inc_json': inc_json,
        'inc_pct_json':inc_pct_json,
        'inc_qq_json':inc_qq_json,
        'inc_ranks_json':inc_ranks_json,
        'inc_peer_json':inc_peer_json,

        'bs_json':bs_json,
        'bs_pct_json':bs_pct_json,
        'bs_ranks_json':bs_ranks_json,
        'bs_peer_json':bs_peer_json,

        'cf_json':cf_json,
        'cf_pct_json':cf_pct_json,
        'cf_ranks_json':cf_ranks_json,
        'cf_peer_json':cf_peer_json,

    }

    return render(request, 'fundamentals.html', context)



def sector_performance(request):

    engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
    cnxn = engine.connect()
    data = pd.read_sql(f"select * from EqSectorIxPerf", cnxn).set_index('index')
    data = data.loc[~(data==0).all(axis=1)]
    print(data)

    one_day = ( data.iloc[-1] / data.iloc[-2]) - 1
    one_week = ( data.iloc[-1] / data.iloc[-7]) - 1
    one_month = ( data.iloc[-1] / data.iloc[-30]) - 1
    two_months = ( data.iloc[-1] / data.iloc[-40]) - 1
    three_months = ( data.iloc[-1] / data.iloc[-60]) - 1
    ytd = ( data.iloc[-1] / data.iloc[0]) - 1  # 252

    df = pd.concat([one_day, one_week, one_month, two_months, three_months, ytd], axis = 1).T
    df.index = ['1D', '1W', '1M','2M', '3M', 'YTD']
    df = df.multiply(100)
    
    print(df)

    sector_performance1d = webtools.df_to_highcharts_clustered_bar(df.loc['1D'].to_frame())
    sector_performance1w = webtools.df_to_highcharts_clustered_bar(df.loc['1W'].to_frame())
    sector_performance1m = webtools.df_to_highcharts_clustered_bar(df.loc['1M'].to_frame())
    sector_performance3m = webtools.df_to_highcharts_clustered_bar(df.loc['3M'].to_frame())
    sector_performanceYtd = webtools.df_to_highcharts_clustered_bar(df.loc['YTD'].to_frame())

    context = {

        'sector_performance1d':sector_performance1d,
        'sector_performance1w':sector_performance1w,
        'sector_performance1m':sector_performance1m,
        'sector_performance3m':sector_performance3m,
        'sector_performanceYtd':sector_performanceYtd,
        
    }

    return render(request, "sector_performance.html", context)



def attribution(request):
    pass