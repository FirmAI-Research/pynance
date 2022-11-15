from django.shortcuts import render

from django.templatetags.static import static
from pathlib import Path
import sys, os

proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(proj_root, 'lib', 'equity'))

from fundamentals import Fundamentals, Ranks, DCF, Columns
import json

def fundamentals(request):


    ticker = 'AMZN'

    fun = Fundamentals( ticker = ticker)

    """ 
    ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
    │     Income Statement                                                                                                │
    └─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    fun.get( columns = Columns.INCOME.value, limit = 5 )

    inc = fun.df.divide(1000000).T.reset_index(level=[0,1]).reset_index(drop=False).drop(columns = ['ticker', 'index'])
    inc.columns.name = None

    print(inc)
    # Convert pandas dataframe to data and column objects read by jquery datatables
    inc_json = inc.to_json(orient='split', index=False)
    j = json.loads(json.dumps(inc_json ))
    print(j)
    inc_data = json.dumps(json.loads(j)["data"])
    # inc_data = json.dumps(inc_data )
    print(inc_data)
    # columns = j['columns']

    context = {

        'table_data': inc_data,

    }

    return render(request, 'fundamentals.html', context)

