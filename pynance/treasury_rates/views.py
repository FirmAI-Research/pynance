from calendar import week
from django.shortcuts import render

from lib.fixed_income.rates.treasury_rates import TreasuryRates

def index(request):

    tr = TreasuryRates()
    rates_table = tr.get()   # FIXME: format for use w HighCharts.fromJSON()
    rates_table.date = rates_table.date.apply(lambda x : x.strftime('%Y-%m-%d'))

    points_in_time_json = tr.point_in_time_curves()

    all_tenors_over_time, all_tenors_x_axis = tr.all_tenors_over_time()
    
    weekly_change, change_x_axis = tr.change_distribution()

    weekly_change_spider, change_x_axis_spider = tr.change_distribution_spider()


    context = {
        
        'rates_table':rates_table,
        'values':rates_table.values.tolist(),

        'points_in_time_json':points_in_time_json,

        'all_tenors_over_time':all_tenors_over_time,
        'all_tenors_x_axis':all_tenors_x_axis,

        'weekly_change':weekly_change,
        'change_x_axis':change_x_axis,

        'weekly_change_spider':weekly_change_spider,
        'change_x_axis_spider':change_x_axis_spider

    }

    return render(request, 'treasury_rates.html', context)