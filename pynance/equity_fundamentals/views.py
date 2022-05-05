from django.shortcuts import render
from matplotlib.font_manager import json_dump
import pandas as pd
import sys
import os
import json
import nasdaqdatalink
from lib.nasdaq import Fundamentals, Metrics, Tickers, Nasdaq
from lib.calendar import Calendar
cal = Calendar()
from dateutil.relativedelta import relativedelta
import datetime
from tabulate import tabulate

cwd = os.getcwd()

iodir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))) + '/_tmp/nasdaq_data_link/'

def dcf(request):
    # select a company from the sector view to load the dcf view
    # dcf --> dcf view
    ticker = 'AMZN'

    ndq = Nasdaq()
    ndq.authenticate()
    ndq_data = nasdaqdatalink.get_table('SHARADAR/SF1',  dimension = 'MRQ', ticker = ticker) # calendardate=cal.previous_quarter_end()

    qtr_end_dates = cal.quarter_end_list(start_date=datetime.datetime.now() - relativedelta(years=2), end_date=cal.today())
    future_qtr_end_dates = cal.quarter_end_list(start_date=cal.today(), end_date=datetime.datetime.now() + relativedelta(years=2))

    def write_to_json_for_ajax():
        ''' based on user selection of parameter values write to ajax json data file'''
        fp = os.path.join(cwd, 'equity_fundamentals', 'static', 'opperating_income.json') # TODO; iterate through all files in the static folder so long as same structure
        print(fp)
        with open(fp, 'r') as f:
            data = json.load(f)

        y = ndq_data.loc[ndq_data.calendardate == qtr_end_dates[-1]]
        yminus1 = ndq_data.loc[ndq_data.calendardate == qtr_end_dates[-2]]
        yminus2 = ndq_data.loc[ndq_data.calendardate == qtr_end_dates[-3]]
        yminus3 = ndq_data.loc[ndq_data.calendardate == qtr_end_dates[-4]]
        yminus4 = ndq_data.loc[ndq_data.calendardate == qtr_end_dates[-5]]

        for i in range(len(data['data'])): # iterate thrugh each dictionary in the list
            source_name= data['data'][i].get('source_name')
            if source_name != '':    
                data['data'][i]['y'] = "{: ,}".format(y[source_name].iloc[0])
                data['data'][i]['y-1'] = "{: ,}".format(yminus1[source_name].iloc[0])
                data['data'][i]['y-2'] = "{: ,}".format(yminus2[source_name].iloc[0])
                data['data'][i]['y-3'] = "{: ,}".format(yminus3[source_name].iloc[0])
                data['data'][i]['y-4'] = "{: ,}".format(yminus4[source_name].iloc[0])

                data['data'][i]['delta1'] = "{:.2%}".format((y[source_name].iloc[0] - yminus1[source_name].iloc[0]) / yminus1[source_name].iloc[0])
                data['data'][i]['delta2'] = "{:.2%}".format((yminus1[source_name].iloc[0] - yminus2[source_name].iloc[0]) / yminus2[source_name].iloc[0])
                data['data'][i]['delta3'] = "{:.2%}".format((yminus2[source_name].iloc[0] - yminus3[source_name].iloc[0]) / yminus3[source_name].iloc[0])
                data['data'][i]['delta4'] = "{:.2%}".format((yminus3[source_name].iloc[0] - yminus4[source_name].iloc[0]) / yminus4[source_name].iloc[0])

        json_dump(data, fp)

    write_to_json_for_ajax()

    context = {
        'qtr_end_dates': qtr_end_dates[-5:],
        'future_qtr_end_dates':future_qtr_end_dates[:4],

    }
    return render(request, 'dcf.html', context)





def fundamentals(request):
    ticker = request.POST.get("ticker")
    print(ticker)

    if ticker in [None, '', '[None]']:
        ticker = 'JPM'

    print(ticker)

    tick = Tickers().get() # FIXME Tickers.get() is bein called twice - Once directly, and once as part of the Fundamentals class
    ticker_row = tick.loc[tick.ticker == ticker]
    sector = ticker_row.sector.iloc[0]
    calendardate = cal.prior_quarter_end()

    if ticker:
        # Quarterly Fundamental data
        fun = Fundamentals(calendardate=calendardate)
        df = fun.fundamentals_by_sector(sector=sector)
        df = fun.view_sector()
        print(df.head())

        # ROE
        qs, outliers = fun.calculate_box_plot(df, column='roe')
        qs_json = json.dumps([float(q) for q in qs])
        outliers_json = json.dumps([float(q) for q in outliers])

        # PE
        qs2, outliers2 = fun.calculate_box_plot(df, column='pe')
        qs_json2 = json.dumps([float(q) for q in qs2])
        outliers_json2 = json.dumps([float(q) for q in outliers2])

        try:
            selected_row_data = df.loc[df['ticker'] == ticker]
            selected = selected_row_data['roe'].values[0] # 0 preceding selected_pe refers to the column position to plot onto in the chart
            selected_json = json.dumps([0, float(selected)])
            selected2 = selected_row_data['pe'].values[0]
            selected_json2 = json.dumps([0, float(selected2)])
        except:
            selected_json = json.dumps([0, 0])
            selected_json2 = json.dumps([0, 0])

        # Quartiles of the Sector
        pctile_frame = fun.build_percentiles_frame(df)
        pctile_frame = pctile_frame.reset_index()
        print(pctile_frame)

        # Selected Company raw data
        ticker_data = df.loc[df.ticker == ticker]
        ticker_data = ticker_data[[x for x in fun.fundamental_cols]]
        for c in ticker_data.columns:
            try:
                ticker_data[c] = ticker_data[c].apply(
                    lambda x: '{:,.2f}'.format(x))
            except Exception:
                pass
        ticker_data.drop(columns=['calendardate'], inplace=True)
        ticker_data.reset_index()

        # All companies in the sector
        all_sector_data = df
        all_sector_data = all_sector_data[[x for x in fun.fundamental_cols]]
        for c in ticker_data.columns:
            if c not in  ['ticker', 'calendardate']:
                all_sector_data[c] = all_sector_data[c].apply(lambda x: '{:,.2f}'.format(x))
        all_sector_data.drop(columns=['calendardate'], inplace=True)
        all_sector_data.reset_index()
        all_companies_in_sector = all_sector_data

        # Selected Company percentile rank for each column
        all_sector_data = df
        all_sector_data = all_sector_data[[ x for x in fun.fundamental_cols] + ['ticker']]
        frames = []
        for c in all_sector_data.columns:
            if c != 'ticker':
                frames.append(all_sector_data[c].rank(pct=True, ascending = False))
        company_pct_rank_data = pd.concat(frames, axis=1)
        company_pct_rank_data['ticker'] = all_sector_data['ticker'].iloc[:, :1]
        company_pct_rank_data = company_pct_rank_data.loc[company_pct_rank_data.ticker == ticker]
        company_pct_rank_data.drop(columns = ['calendardate'], inplace = True)
        cols = company_pct_rank_data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        company_pct_rank_data = company_pct_rank_data[cols]
        for c in company_pct_rank_data.columns:
            if c != 'ticker':
                company_pct_rank_data[c] = company_pct_rank_data[c].apply(lambda x : '{:,.2f}'.format(x))
        print(company_pct_rank_data)

        context = {
            'selected_ticker': ticker,
            'sector': sector,
            'calendardate':calendardate,

            'qs_json': qs_json,
            'selected_pe_json': selected_json,

            'qs_json2': qs_json2,
            'selected_pe_json2': selected_json2,

            'pctile_frame': pctile_frame,
            'values': pctile_frame.values.tolist(),

            'ticker_data': ticker_data,
            'ticker_values': ticker_data.values.tolist(),

            'all_sector_data': all_companies_in_sector,
            'sector_values': all_companies_in_sector.values.tolist(),

            'company_pct_rank_data': company_pct_rank_data,
            'company_pct_rank_values': company_pct_rank_data.values.tolist(),
        }

    else:
        context = {}

    return render(request, 'fundamentals.html', context)



