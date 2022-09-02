#  python -m pip install QuantLib
import QuantLib as ql


class Pricing:

    def __init__(self) -> None:

        # Everything starts with “evaluation date” which means the date you want to value a instrument.
        today = ql.Date(15,8,2022)
        ql.Settings.instance().evaluationDate = today
        print(ql.Settings.instance().evaluationDate)

        calendar = ql.UnitedStates()

    def cash_flow(self):
        date = ql.Date().todaysDate()
        cf1 = ql.SimpleCashFlow(5.0, date+365)
        cf2 = ql.SimpleCashFlow(5.0, date+365*2)
        cf3 = ql.SimpleCashFlow(105.0, date+365*3)
        leg = ql.Leg([cf1, cf2, cf3])

        amount = 105
        date = ql.Date(15,6,2020)
        cf = ql.SimpleCashFlow(amount, date)

        yts = ql.YieldTermStructureHandle(ql.FlatForward(ql.Date(15,1,2020), 0.04, ql.Actual360()))
        ql.CashFlows.npv(leg, yts, True)
        ql.CashFlows.npv(leg, yts, True, ql.Date(15,6,2020))
        ql.CashFlows.npv(leg, yts, True, ql.Date(15,6,2020), ql.Date(15,12,2020))


    def bond(self):
        start = ql.Date(15,12,2022)
        maturity = ql.Date(15,12,2023)

        schedule = ql.MakeSchedule(start, maturity, ql.Period('6M'))
        interest = ql.FixedRateLeg(schedule, ql.Actual360(), [100.], [0.05])
        
        bond = ql.Bond(0, ql.TARGET(), start, interest)
        
        print(bond.bondYield(100, ql.Actual360(), ql.Compounded, ql.Annual))
        print(bond.dirtyPrice(0.05, ql.Actual360(), ql.Compounded, ql.Annual))

        return bond


    def fixed_rate_bond(self):
        bond = ql.FixedRateBond(2, ql.TARGET(), 100.0, ql.Date(15,12,2019), ql.Date(15,12,2024), ql.Period('1Y'), [0.05], ql.ActualActual())


    def amortizing_fixed_rate_bond(self):
        notionals = [100,100,100,50]
        schedule = ql.MakeSchedule(ql.Date(25,1,2018), ql.Date(25,1,2022), ql.Period('1y'))
        bond = ql.AmortizingFixedRateBond(0, notionals, schedule, [0.03], ql.Thirty360())


    def floating_rate_bond(self):
        schedule = ql.MakeSchedule(ql.Date(15,6,2020), ql.Date(15,6,2022), ql.Period('6m'))
        index = ql.Euribor6M()
        ql.FloatingRateBond(2,100, schedule, index, ql.Actual360(), spreads=[0.01])


    def callable_bond(self):
        schedule = ql.MakeSchedule(ql.Date(15,6,2020), ql.Date(15,6,2022), ql.Period('1Y'))
        putCallSchedule = ql.CallabilitySchedule()

        callability_price  = ql.CallabilityPrice(100, ql.CallabilityPrice.Clean)

        putCallSchedule.append(
            ql.Callability(callability_price, ql.Callability.Call, ql.Date(15,6,2021))
        )

        ql.CallableFixedRateBond(2, 100, schedule, [0.01], ql.Actual360(), ql.ModifiedFollowing, 100, ql.Date(15,6,2020), putCallSchedule)


    def zero_coupon_bond(self):
        bond = ql.ZeroCouponBond(2, ql.TARGET(), 100, ql.Date(20,6,2023))

        print(bond.dirtyPrice(0.05, ql.Actual360(), ql.Compounded, ql.Annual))

        return bond



