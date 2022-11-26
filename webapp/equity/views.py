from django.shortcuts import render

from django.templatetags.static import static
from pathlib import Path
import sys, os
import json
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(proj_root, 'lib'))
sys.path.append(os.path.join(proj_root, 'lib', 'equity'))

from fundamentals import Fundamentals, Ranks, DCF, Columns
import webtools

fp_ajax = os.path.join( Path(__file__).resolve().parent, 'ajax')



def fundamentals(request):


    ticker = 'AMZN'

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

    rank = Ranks(ticker = ticker)
    inc_ranks = rank.get_ranks().for_js(cols=Columns.INCOME.value)
    fp = os.path.join(fp_ajax, 'income_statement_ranks.json')
    inc_ranks_json = webtools.df_to_dt(inc_ranks, fp)

    peers = fun.get_peers()
    peer_fun = Fundamentals(ticker = peers)
    inc_peer_values =  peer_fun.get( columns = Columns.INCOME.value, limit = 5 ).for_js(multiIndex = True)
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
    bs_ranks = rank.get_ranks().for_js(cols=Columns.BALANCE.value)
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
    cf_ranks = rank.get_ranks().for_js(cols=Columns.CASHFLOW.value)
    fp = os.path.join(fp_ajax, 'cash_flow_ranks.json')
    cf_ranks_json = webtools.df_to_dt(cf_ranks, fp)

    peers = fun.get_peers()
    peer_fun = Fundamentals(ticker = peers)
    cf_peer_values =  peer_fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).for_js(multiIndex = True)
    fp = os.path.join(fp_ajax, 'cash_flow_peer_values.json')    
    cf_peer_json =  webtools.df_to_dt(cf_peer_values.fillna('-'), fp)


    context = {

        'inc_json': inc_json,
        'inc_pct_json':inc_pct_json,
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

