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

    inc = fun.df.divide(1000000).T.droplevel(1)

    print(inc)
    # Convert pandas dataframe to data and column objects read by jquery datatables
    inc_json = inc.to_json(orient='split', index=False)
    print(type(inc_json))
    j = json.loads(inc_json)
    inc_data = j['data']
    columns = j['columns']
    columns = [{'title': c} for c in columns]
    print(inc_data)
    print(columns)
    table_data = inc.to_html(table_id="example")

    context = {

        'table_data': table_data,

    }

    return render(request, 'fundamentals.html', context)

