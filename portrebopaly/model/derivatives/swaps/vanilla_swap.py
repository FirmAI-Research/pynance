import QuantLib as ql
import pandas as pd 

def init():
    yts = ql.RelinkableYieldTermStructureHandle()

    instruments = [
        ('depo', '6M', 0.025),
        ('fra', '6M', 0.03),
        ('swap', '2Y', 0.032),
        ('swap', '3Y', 0.035)
    ]

    helpers = ql.RateHelperVector()
    index = ql.Euribor6M(yts)
    for instrument, tenor, rate in instruments:
        if instrument == 'depo':
            helpers.append( ql.DepositRateHelper(rate, index) )
        if instrument == 'fra':
            monthsToStart = ql.Period(tenor).length()
            helpers.append( ql.FraRateHelper(rate, monthsToStart, index) )
        if instrument == 'swap':
            swapIndex = ql.EuriborSwapIsdaFixA(ql.Period(tenor))
            helpers.append( ql.SwapRateHelper(rate, swapIndex))
    curve = ql.PiecewiseLogCubicDiscount(2, ql.TARGET(), helpers, ql.ActualActual())
    yts.linkTo(curve)
    engine = ql.DiscountingSwapEngine(yts)
    tenor = ql.Period('2y')
    fixedRate = 0.05
    forwardStart = ql.Period("2D")

    swap = ql.MakeVanillaSwap(tenor, index, fixedRate, forwardStart, Nominal=10e6, pricingEngine=engine)
    airRate = swap.fairRate()
    npv = swap.NPV()
    #print(f"Fair swap rate: {fairRate:.3%}")
    print(f"Swap NPV: {npv:,.3f}")

    import pandas as pd
    pd.options.display.float_format = "{:,.2f}".format

    cashflows = pd.DataFrame({
        'date': cf.date(),
        'amount': cf.amount()
        } for cf in swap.leg(1))
    print(cashflows)

    cashflows = pd.DataFrame({
        'nominal': cf.nominal(),
        'accrualStartDate': cf.accrualStartDate().ISO(),
        'accrualEndDate': cf.accrualEndDate().ISO(),
        'rate': cf.rate(),
        'amount': cf.amount()
        } for cf in map(ql.as_coupon, swap.leg(1)))
    print(cashflows)


if __name__ == '__main__':
    init()