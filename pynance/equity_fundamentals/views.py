from django.shortcuts import render
import pandas as pd
import sys
import os
import json

from lib.nasdaq import Fundamentals, Metrics, Tickers
from lib.calendar import Calendar
cal = Calendar()

iodir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))) + '/_tmp/nasdaq_data_link/'


def fundamentals(request):
    ticker = request.POST.get("ticker")
    print(ticker)

    if ticker in [None, '', '[None]']:
        ticker = 'JPM'

    print(ticker)

    tick = Tickers().get()
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
            try:
                all_sector_data[c] = all_sector_data[c].apply(
                    lambda x: '{:,.2f}'.format(x))
            except Exception:
                pass
        all_sector_data.drop(columns=['calendardate'], inplace=True)
        all_sector_data.reset_index()

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

            'all_sector_data': all_sector_data,
            'sector_values': all_sector_data.values.tolist(),

            'company_pct_rank_data': company_pct_rank_data,
            'company_pct_rank_values': company_pct_rank_data.values.tolist(),
        }

    else:
        context = {}

    return render(request, 'fundamentals.html', context)





def dcf(request):
    # select a company from the sector view to load the dcf view
    # dcf --> dcf view
    context = {

    }
    return render(request, 'dcf.html', context)

