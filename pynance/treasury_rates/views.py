from django.shortcuts import render

from lib.fixed_income.rates.treasury_rates import TreasuryRates

def index(request):

    tr = TreasuryRates()
    rates_table = tr.get()   # FIXME: format for use w HighCharts.fromJSON()
    rates_table.date = rates_table.date.apply(lambda x : x.strftime('%Y-%m-%d'))
    print(rates_table)

    points_in_time_json = tr.point_in_time_curves()
    print(points_in_time_json)

    #df = df.reindex(index=df.index[::-1]) # reverse the dataframe

    context = {
        
        'rates_table':rates_table,
        'values':rates_table.values.tolist(),

        'points_in_time_json':points_in_time_json

    }

    return render(request, 'treasury_rates.html', context)