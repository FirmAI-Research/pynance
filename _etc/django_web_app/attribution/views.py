import sys
import os
import pandas as pd
from django.views.decorators.csrf import csrf_protect 

from django.shortcuts import render

# from lib.attribution.Famma_French.famma_french import FammaFrench
from lib.attribution.Famma_French.famma_french_II import FammaFrench
from lib.attribution.PCA.portfolio_risk_pca import PCA_VaR, VaR

from lib.calendar import Calendar
cal = Calendar()

def index(request):
    print('asdf')

    return render(request, 'attribution_index.html')


@csrf_protect 
def famma_french(request):
    
    # pca = PCA_VaR()
    # pca = VaR()

    tickers = str(request.POST.get("tickers")).split(', ')
    tickers = ['IWM'] if tickers == ['None'] else tickers
        
    weights = str(request.POST.get("weights")).split(', ')
    weights = [1] if weights == ['None'] else weights
    weights = [float(i) for i in weights]

    print(tickers)
    print(weights)
    ff = FammaFrench(tickers, weights)

    context = {
        "model_summary": ff.model_summary.as_html().replace('<table class="simpletable">', '').replace('</table>', '',),
        'tickers': ", ".join(tickers),
        'weights': ", ".join([str(weight) for weight in weights]),
        # 'model_name': model_name,
        'statements': ff.statements,
        'date_start': ff.START_DATE,
    }


    return render(request, 'famma_french.html', context=context)

