from django.shortcuts import render
from matplotlib.font_manager import json_dump
import pandas as pd
import sys
import os
import json
import nasdaqdatalink
from pyparsing import line
from lib.nasdaq import Fundamentals, Metrics, Tickers, Nasdaq
from lib.calendar import Calendar
from db.postgres import Postgres
from lib import numeric

cal = Calendar()
from dateutil.relativedelta import relativedelta
import datetime
from tabulate import tabulate

cwd = os.getcwd()

iodir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))) + '/_tmp/nasdaq_data_link/'



def financials(request):

    ticker = str(request.POST.get("tickers"))
    if ticker in ['', 'None', None]:
        ticker = 'AMZN'
    print(ticker)
    print(request.POST)

    ndq_data = Fundamentals(ticker = ticker).get()
    print(ndq_data)
    qtr_end_dates = cal.quarter_end_list(start_date=datetime.datetime.now() - relativedelta(years=2), end_date=cal.today())

    def write_to_json_for_ajax():
        ''' write values to json files used to populate jquery datatables via ajax '''
        
        # collect paths to json file for each datatable
        fp1 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'opperations.json')
        fp2 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'adjustments.json')

        # iterate through each file that needs to be writen to
        for fp in [fp1, fp2]:
            with open(fp, 'r') as f:
                data = json.load(f)

            # build dictionary with keys == date and values == pd.series of fundamental equity data for that specific date
            fundamentals_dict = {}
            for date in qtr_end_dates[-5:]:
                fundamentals_dict[date] = ndq_data.loc[ndq_data.calendardate == date]

            # iterate through each dictionary item and write values to the desired node of the json file
            ajax_labels =[  'y-4', 'y-3', 'y-2', 'y-1','y',]
            ajax_delta_labels = ['delta5', 'delta4','delta3','delta2','delta1']
            for i in range(len(data['data'])): 
                source_name= data['data'][i].get('source_name')
                for ix, label in enumerate(ajax_labels):
                    if source_name != '':    
                        data['data'][i][label] = "{: ,}".format(list(fundamentals_dict.values())[ix][source_name].iloc[0])
                    else:
                        data['data'][i][label] = "None"

                for ix, label in enumerate(ajax_delta_labels):
                    if label != 'delta5':
                        try:
                            data['data'][i][label] = "{:.2%}".format((list(fundamentals_dict.values())[ix][source_name].iloc[0] - list(fundamentals_dict.values())[ix-1][source_name].iloc[0]) / list(fundamentals_dict.values())[ix-1][source_name].iloc[0])
                        except KeyError as e:
                            pass
            json_dump(data, fp)

    write_to_json_for_ajax()

    context = {
        'qtr_end_dates': qtr_end_dates[-5:],
        'ticker': ticker,
        'as_of_date':ndq_data.lastupdated.iloc[0]
    }

    return render(request, 'financials.html', context)




def fundamentals(request):
    ticker = request.POST.get("ticker")

    if ticker in [None, '', '[None]']:
        ticker = 'AMZN'
    print(ticker)
    print(request.POST)

    ndq = Nasdaq()
    ndq.authenticate()

    # boxplots
    ticker_data = nasdaqdatalink.get_table('SHARADAR/TICKERS', ticker = ticker)
    industry = ticker_data['industry'].iloc[0]
    industry_of_selected_ticker = industry.replace(' ','_')
    sector = ticker_data['sector'].iloc[0]

    data_of_selected_company = pd.DataFrame(Fundamentals(ticker = ticker).get().iloc[0]).transpose()
    data_of_selected_company = data_of_selected_company[['calendardate', 'netinc', 'equity', 'debt','assets','pe','pb','ps', 'evebit', 'ebit', 'divyield', 'marketcap', 'ev', 'ebitda', 'fcf','opinc','revenue', 'intexp', 'roe']]

    calendardate = pd.to_datetime(data_of_selected_company['calendardate'].values[0]).strftime('%Y-%m-%d')

    engine = Postgres().engine

    # boxplots
    metric_list = ['pe', 'roe']
    box_plot_values = []
    company_values = []

    for metric in metric_list:
        # Sector
        sector_percentiles = pd.read_sql_table('Percentiles_Sector', engine)  # a list of values representing the min, max, median, 1st and 3rd quartile
        sector_percentiles = sector_percentiles.loc[sector_percentiles.sector == sector]
        sector_percentiles_values = [float(x) for x in sector_percentiles[metric]]
        # Industry
        industry_percentiles = pd.read_sql_table(f'Percentiles_Industry', engine)
        industry_percentiles = industry_percentiles.loc[industry_percentiles.industry == industry]
        industry_percentiles_values = [float(x) for x in industry_percentiles[metric]]
        # Values
        box_plot_values.append( [sector_percentiles_values] + [industry_percentiles_values] )
        company_values.append(   [data_of_selected_company[metric].values[0] ]  + [data_of_selected_company[metric].values[0] ]  )
        print(box_plot_values)
        print(company_values)

    for c in [x for x in data_of_selected_company.columns if x != 'calendardate']:
        data_of_selected_company[c] = "{:,}".format(float(data_of_selected_company[c]))


    # industry percetiles v. metric over time
    metric_list = ['pe', 'roe']

    data = Fundamentals(ticker = ticker).get()

    line_chart_json_list = []

    for metric in metric_list:
        metric_data = data[['calendardate', metric]]
        metric_data['calendardate'] = [x.strftime('%Y%m%d') for x in metric_data['calendardate']]
        
        frames = []
        for k,v in {'Quartile_Values_Over_Time_Median_by_Industry':'Median','Quartile_Values_Over_Time_Upper_by_Industry':'Upper','Quartile_Values_Over_Time_Lower_by_Industry':"Lower"}.items():
            x = pd.read_sql_table(k, engine) 
            x = x[['date',metric]].loc[x.industry == industry].rename(columns={'date':'calendardate', metric:v})
            x['calendardate'] = [ pd.to_datetime(x).strftime('%Y%m%d') for x in x['calendardate']]
            frames.append(x)
        cbind = metric_data

        for df in frames:
            cbind = cbind.merge(df, on ='calendardate')

        for c in cbind:
            cbind[c] = cbind[c].apply(lambda x: float(x))

        cbind.drop('calendardate', axis=1, inplace=True)
        linechart_data = ndq.to_highcharts(cbind)
        linechart_data = json.loads(linechart_data)
        for i in range(len(linechart_data)):
            if i > 0:
                linechart_data[i]["color"] =  '#D3D3D3'
        line_chart_json_list.append(json.dumps(linechart_data))
    print(line_chart_json_list[1])




    context = {

        'selected_ticker': ticker,
        'sector': sector,
        'industry':industry,
        'calendardate':calendardate,
        # boxplots
        'box_plot_values_1': box_plot_values[0],
        'selected_company_values_1': company_values[0],
        'box_plot_values_2': box_plot_values[1],
        'selected_company_values_2': company_values[1],
        # industry percentiles over time
        'line_chart_values_1':line_chart_json_list[0],
        'line_chart_values_2':line_chart_json_list[1],


        'company_fundamentals':data_of_selected_company,
        'values':data_of_selected_company.values.tolist(),

    }


    return render(request, 'fundamentals.html', context)



