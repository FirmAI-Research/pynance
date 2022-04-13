from django.shortcuts import render

from lib.fixed_income.rates.treasury_rates import TreasuryRates

def index(request):

    tr = TreasuryRates()
    df = tr.get()   # FIXME: format for use w HighCharts.fromJSON()
    df.date = df.date.apply(lambda x : x.strftime('%Y-%m-%d'))
    print(df)

    print(df.dtypes)

    #df = df.reindex(index=df.index[::-1]) # reverse the dataframe

    context = {
        
        'df':df,
        'values':df.values.tolist(),

    }

    return render(request, 'treasury_rates.html', context)