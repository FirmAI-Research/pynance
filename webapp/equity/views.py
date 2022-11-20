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
    inc_pct_json = fun.pct_chg.T.reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
    inc_pct_json.columns.name = None
    fp = os.path.join(fp_ajax, 'income_statement_pct.json')
    inc_pct_json = webtools.df_to_dt(inc_pct_json.fillna('-'), fp)

    rank = Ranks(ticker = ticker)
    ranks = rank.get_ranks() 
    inc_ranks = rank.rank_pivot
    inc_ranks.columns.name = None
    inc_ranks.index.name = None
    inc_ranks = inc_ranks.droplevel(0, axis=1)
    print(inc_ranks)


    """ 
    ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │ Cash Flow                                                                                                        │
    └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    cf = fun.get( columns = Columns.CASHFLOW.value, limit = 5 ).for_js()
    fp = os.path.join(fp_ajax, 'cash_flow.json')
    cf_json = webtools.df_to_dt(cf, fp)



    context = {

        'inc_json': inc_json,
        'inc_pct_json':inc_pct_json,
        'cf_json': cf_json,


    }

    return render(request, 'fundamentals.html', context)

