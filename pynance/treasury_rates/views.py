from django.shortcuts import render

from lib.fixed_income.rates.treasury_rates import TreasuryRates

def index(request):

    tr = TreasuryRates()
    df = tr.get()   # FIXME: format for use w HighCharts.fromJSON()
    print(df)

    context = {
        
        'df':df,
        'values':df.values.tolist(),

    }

    return render(request, 'treasury_rates.html', context)