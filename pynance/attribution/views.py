import sys, os

from django.shortcuts import render

from lib.attribution.Famma_French.famma_french import FammaFrench


def index(request):
    print('asdf')

    return render(request, 'attribution_FF.html')


def famma_french(request):

    io_dirp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) ) # NOTE: relative path from manage.py to lib/
    sys.path.append(io_dirp) 

    ff = FammaFrench(symbols = ['SPY'],  weights = [1], iodir=os.path.join(io_dirp, '_tmp/ff/'))
    ff.merge_factors_and_portfolio(download_ff_data=True)    # TODO store ff download in db table each morning
    ff.three_factor()

    model_summary = ff.model.summary().as_html().replace('<table class="simpletable">', '').replace('</table>', '')
    print(model_summary)

    context = { 

        'model_summary': model_summary,

    }
    return render(request, 'attribution_FF.html', context=context)