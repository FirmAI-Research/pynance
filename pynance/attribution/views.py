import sys
import os
import pandas as pd
from django.views.decorators.csrf import csrf_protect 

from django.shortcuts import render

# from lib.attribution.Famma_French.famma_french import FammaFrench
from lib.attribution.Famma_French.famma_french_II import FammaFrench

from lib.calendar import Calendar
cal = Calendar()

def index(request):
    print('asdf')

    return render(request, 'attribution_index.html')


@csrf_protect 
def famma_french(request):

    tickers = str(request.POST.get("tickers")).split(', ')
    tickers = ['IWM'] if tickers == ['None'] else tickers
        
    weights = str(request.POST.get("weights")).split(', ')
    weights = [1] if weights == ['None'] else weights
    weights = [float(i) for i in weights]

    # model_name = str(request.POST.get("model_name"))
    # if model_name == 'None':
    #     model_name = 'three_factor'

    print(tickers)
    print(weights)
    ff = FammaFrench(tickers, weights)

    context = {
        "model_summary": ff.model_summary.as_html().replace('<table class="simpletable">', '').replace('</table>', '',),
        'tickers': ", ".join(tickers),
        'weights': ", ".join([str(weight) for weight in weights]),
        # 'model_name': model_name,
        'date_start': ff.START_DATE,
    }

    return render(request, 'famma_french.html', context=context)



# @csrf_protect 
# def famma_french(request):

#     io_dirp = os.path.dirname(os.path.dirname(os.path.dirname(
#         os.path.abspath(__file__))))  # NOTE: relative path from manage.py to lib/
#     sys.path.append(io_dirp)

#     # post parameters
#     x = submitbutton = request.POST.get("submit")
#     tickers = str(request.POST.get("tickers")).split(', ')
#     weights = str(request.POST.get("weights")).split(', ')
#     model_name = str(request.POST.get("model_name"))

#     try: 
#         date_start = pd.to_datetime(str(request.POST.get("datepicker_start"))).strftime('%Y-%m-%d')
#     except Exception:
#         date_start = '2022-01-01'


#     try: 
#         date_end = str(request.POST.get("datepicker_end"))
#     except Exception:
#         date_end = cal.today()


#     print(model_name)
#     print(tickers)
#     print(weights)
#     print(date_start)
#     print(date_end)

#     if tickers == ['None']:
#         tickers = ['VWO']
#     if weights == ['None']:
#         weights = [1]
#     if model_name == 'None':
#         model_name = 'three_factor'

#     # run models
#     ff = FammaFrench(symbols=tickers,  weights=weights, date_start = date_start, date_end=None, iodir=os.path.join(io_dirp, '_tmp/ff/'))
#     # TODO store ff download in db table each morning
#     ff.merge_factors_and_portfolio(download_ff_data=True)
#     if model_name == 'three_factor':
#         ff.three_factor()
#     elif model_name == 'four_factor':
#         ff.four_factor()
#     elif model_name == 'five_factor':
#         ff.five_factor()

#     model_summary = ff.model.summary().as_html().replace(
#         '<table class="simpletable">', '').replace('</table>', '')

#     context = {

#         'model_summary': model_summary,
#         'tickers': tickers,
#         'weights': weights,
#         'model_name': model_name,
#         'date_start': date_start,

#     }
#     return render(request, 'attribution_FF.html', context=context)
