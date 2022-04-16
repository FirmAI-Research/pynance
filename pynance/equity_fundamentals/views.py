from django.shortcuts import render
import pandas as pd 
import sys, os 
import json 

from lib.nasdaq import Fundamentals, Metrics
from lib.calendar import Calendar
cal = Calendar()

iodir =  os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/_tmp/nasdaq_data_link/'  

def fundamentals(request):
    # index --> sector view
    
    selected_ticker = request.POST.get("tickers")

    if selected_ticker is None or selected_ticker == '':
        selected_ticker = 'ZNTE'


    if selected_ticker:

        # Quarterly Fundamental data
        fun = Fundamentals()
        df = fun.fundamentals_by_sector(sector = 'Industrials')
        df = fun.view_sector()
        print(df.head())

        # ROE
        qs, outliers = fun.calculate_box_plot(df, column = 'roe')
        qs_json = json.dumps([float(q) for q in qs])
        outliers_json = json.dumps([float(q) for q in outliers])

        # PE
        qs2, outliers2 = fun.calculate_box_plot(df, column = 'pe')
        qs_json2 = json.dumps([float(q) for q in qs2])
        outliers_json2 = json.dumps([float(q) for q in outliers2])

        try:
            selected_row_data = df.loc[df['ticker'] == selected_ticker]

            selected = selected_row_data['roe'].values[0]
            selected_json = json.dumps([0, float(selected)]) # 0 preceding selected_pe refers to the column position to plot onto in the chart
            selected2 = selected_row_data['pe'].values[0]
            selected_json2 = json.dumps([0, float(selected2)]) # 0 preceding selected_pe refers to the column position to plot onto in the chart
        except:
            selected_json = json.dumps([0, 0]) # 0 preceding selected_pe refers to the column position to plot onto in the chart



        pctile_frame = fun.build_percentiles_frame(df)
        print(pctile_frame)
        


        context = {
            'selected_ticker':selected_ticker,
 
            'qs_json':qs_json,
            'selected_pe_json':selected_json,

 
            'qs_json2':qs_json2,
            'selected_pe_json2':selected_json2,

            'pctile_frame':pctile_frame,
            'values':pctile_frame.values.tolist(),
        }
    
    else:
        context = {}
    
    return render(request, 'fundamentals.html', context)


def dcf(request):
    # select a company from the sector view to load the dcf view
    # dcf --> dcf view
    pass