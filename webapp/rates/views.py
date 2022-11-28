from django.shortcuts import render

from django.templatetags.static import static
from pathlib import Path
import sys, os
import json
proj_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(os.path.join(proj_root, 'lib'))
sys.path.append(os.path.join(proj_root, 'lib', 'fixed_income'))

from interest_rates import Treasuries
import webtools

fp_ajax = os.path.join( Path(__file__).resolve().parent, 'ajax')

def treasuries(request):
    print(os.path.join(proj_root, 'lib', 'fixed_income'))

    ust = Treasuries(years = ['2022'])
    print(ust.df)

    recent_rates = ust.df.iloc[-10:]
    recent_rates_response = webtools.df_to_highcharts_heatmap(recent_rates)
    

    recent_rates_change = ust.df.iloc[-10:].diff(axis=0).dropna(axis=0).round(2)
    recent_rates_change_response = webtools.df_to_highcharts_heatmap(recent_rates_change)



    context = {

        'recent_rates_response':recent_rates_response,
        'recent_rates_change_response':recent_rates_change_response

    }

    return render(request, 'treasuries.html', context)