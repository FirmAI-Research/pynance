from lib2to3.pgen2.pgen import DFAState
from django.shortcuts import render
from matplotlib.font_manager import json_dump
import pandas as pd
import sys
import os
import json
import nasdaqdatalink
from pyparsing import line
from lib.learn.regression.regression import Regression
from lib.nasdaq import Fundamentals, Metrics, Tickers, Nasdaq
from lib.calendar import Calendar
from db.postgres import Postgres
from lib import numeric
import yfinance

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

    fun = Fundamentals(ticker = ticker)
    ndq_data = fun.get()
    print(ndq_data)
    qtr_end_dates = cal.quarter_end_list(start_date=datetime.datetime.now() - relativedelta(years=2), end_date=cal.today())

    def write_to_json_for_ajax():
        ''' write values to json files used to populate jquery datatables via ajax '''
        
        # collect paths to json file for each datatable
        #TODO: os.listdir to read files in directory
        fp1 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'opperations.json')
        fp2 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'adjustments.json')
        fp3 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'fcfgrowth.json')
        fp4 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'wacc.json')
        fp4 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'wacc.json')
        fp5 = os.path.join(cwd, 'equity_fundamentals', 'static', 'forAjax', 'expenses.json')


        # iterate through each file that needs to be writen to
        for fp in [fp1, fp2, fp3, fp4, fp5]:
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
                        data['data'][i][label] = "{:,.4f}".format(list(fundamentals_dict.values())[ix][source_name].iloc[0])
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

    # CAPM
    import statsmodels.api as sm
    import yfinance as yf
    import numpy as np
    df = yf.download([ticker, 'SPY'], '2018-01-01')['Adj Close']
    price_change = df.pct_change()
    df = price_change.drop(price_change.index[0])
    y = df[ticker]
    X = df['SPY']
    model = sm.OLS(y.astype(float), X.astype(float)).fit()
    model_summary = model.summary()
    # produce scatter plot of SPY vs. ticker (pg.54)
    print(model_summary)
    json_scatter = []
    list_values = []
    for key, value in df.iterrows():
        list_values.append([value[0], value[1]])
    json_scatter = json.dumps(list_values)


    context = {
        'qtr_end_dates': qtr_end_dates[-5:],
        'ticker': ticker,
        'as_of_date':ndq_data.lastupdated.iloc[0],
        'beta_values': json_scatter,
    }

    return render(request, 'financials.html', context)




def fundamentals(request):
    ticker = request.POST.get("ticker")
    selected_metric = request.POST.get("metric")
    print(selected_metric)
    if selected_metric in [None, '', '[None]']:
        selected_metric = 'pe'

    if ticker in [None, '', '[None]']:
        ticker = 'AMZN'
    print(ticker)
    print(request.POST)

    engine = Postgres().engine

    ndq = Nasdaq()
    ndq.authenticate()

    # boxplots
    ticker_data = nasdaqdatalink.get_table('SHARADAR/TICKERS', ticker = ticker)
    industry = ticker_data['industry'].iloc[0]
    sector = ticker_data['sector'].iloc[0]
    view_fields = [selected_metric, 'netinc', 'roe','roa','roc', 'pe','pb','ps', 'divyield', 'ev/ebitda', 'p/cf', 'opp margin', 'bvps', 'price','interest coverage', 'payoutratio',]
    
    data_of_selected_company = pd.DataFrame(Fundamentals(ticker = ticker).get().iloc[0]).transpose()
    data_of_selected_company.rename(columns = {'calendardate': 'date'}, inplace=True)

    calendardate = pd.to_datetime(data_of_selected_company['date'].values[0]).strftime('%Y-%m-%d')

    box_plot_values = []
    company_values = []

    for metric in [selected_metric]:
        # Sector
        sector_percentiles = pd.read_sql_table('Percentiles_Sector', engine)  # a list of values representing the min, max, median, 1st and 3rd quartile
        sector_percentiles = sector_percentiles.loc[sector_percentiles.sector == sector]
        sector_percentiles_values = [float(str(x).replace(',','')) for x in sector_percentiles[metric]]

        # Industry
        industry_percentiles = pd.read_sql_table(f'Percentiles_Industry', engine)
        industry_percentiles = industry_percentiles.loc[industry_percentiles.industry == industry]
        industry_percentiles_values = [float(str(x).replace(',','')) for x in industry_percentiles[metric]]
        # Values
        box_plot_values.append( [sector_percentiles_values] + [industry_percentiles_values] )
        company_values.append(   [data_of_selected_company[metric].values[0] ]  + [data_of_selected_company[metric].values[0] ]  )
        print(box_plot_values)
        print(company_values)


    colnames = [x for x in sector_percentiles.columns.tolist() if x not in ['level_0', 'index', 'date','uid']]

    # industry percetiles v. metric over time
    data = Fundamentals(ticker = ticker).get()

    line_chart_json_list = []
    for metric in [selected_metric]:
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
            cbind[c] = cbind[c].apply(lambda x: float(str(x).replace(',','')))

        cbind.drop('calendardate', axis=1, inplace=True)
        linechart_data = ndq.to_highcharts(cbind)
        linechart_data = json.loads(linechart_data)
        for i in range(len(linechart_data)):
            if i > 0:
                linechart_data[i]["color"] =  '#D3D3D3'
        line_chart_json_list.append(json.dumps(linechart_data))


    data_of_selected_company = data_of_selected_company[['date'] + view_fields]
    data_of_selected_company['Peer Group'] = 'N/A'
    data_of_selected_company['Description'] = 'Values as reported'

    for c in [x for x in data_of_selected_company.columns if x not in  ['date', 'Peer Group', 'Description']]:
        try:
            data_of_selected_company[c] = "{:,}".format(float(data_of_selected_company[c]))
        except:
            pass

    industry_ranks_of_selected_company = pd.read_sql_table('Ranks_Industry', engine)
    industry_ranks_of_selected_company = industry_ranks_of_selected_company[['date'] + view_fields].loc[industry_ranks_of_selected_company.ticker == ticker.upper()]
    industry_ranks_of_selected_company['Peer Group'] = 'Industry'
    industry_ranks_of_selected_company['Description'] = 'Percentile Rank against Industry Peer Group'

    sector_ranks_of_selected_company = pd.read_sql_table('Ranks_Sector', engine)
    sector_ranks_of_selected_company = sector_ranks_of_selected_company[['date'] + view_fields].loc[sector_ranks_of_selected_company.ticker == ticker.upper()]
    sector_ranks_of_selected_company['Peer Group'] = 'Sector'
    sector_ranks_of_selected_company['Description'] = 'Percentile Rank against Sector Peer Group'

    percentile_values_of_industry= pd.read_sql_table('Percentiles_Industry', engine)
    percentile_values_of_industry = percentile_values_of_industry[['date'] + view_fields].loc[(percentile_values_of_industry.industry == industry) & (percentile_values_of_industry['index'] =='median')]
    percentile_values_of_industry['Peer Group'] = 'Industry'
    percentile_values_of_industry['Description'] = 'Median Percentile Value amoung Industry Peer Group'

    percentile_values_of_sector = pd.read_sql_table('Percentiles_Sector', engine)
    percentile_values_of_sector = percentile_values_of_sector[['date'] + view_fields].loc[(percentile_values_of_sector.sector == sector) & (percentile_values_of_sector['index'] =='median')]
    percentile_values_of_sector['Peer Group'] = 'Sector'
    percentile_values_of_sector['Description'] = 'Median Percentile Value amoung Sector Peer Group'

    company_fundamentals = pd.concat([data_of_selected_company, percentile_values_of_industry, percentile_values_of_sector, industry_ranks_of_selected_company, sector_ranks_of_selected_company, ], axis=0, ignore_index=True)


    context = {

        'selected_ticker': ticker,
        'sector': sector,
        'industry':industry,
        'calendardate':calendardate,
        'colnames':colnames,
        # boxplots
        'box_plot_values_1': box_plot_values[0],
        'selected_company_values_1': company_values[0],
        # 'box_plot_values_2': box_plot_values[1],
        # 'selected_company_values_2': company_values[1],


        # industry percentiles over time
        'line_chart_values_1':line_chart_json_list[0],
        # 'line_chart_values_2':line_chart_json_list[1],


        'company_fundamentals':company_fundamentals,
        'values':company_fundamentals.values.tolist(),

    }


    return render(request, 'fundamentals.html', context)



