import QuantLib as ql
import pandas as pd 


start = ql.Date(15,12,2019)
maturity = ql.Date(15,12,2025)
schedule = ql.MakeSchedule(start, maturity, ql.Period('6M'))

interest = ql.FixedRateLeg(schedule, ql.Actual360(), [100.], [0.05])
bond = ql.Bond(0, ql.TARGET(), start, interest)
print(bond.bondYield(100, ql.Actual360(), ql.Compounded, ql.Annual))

# print(bond.dirtyPrice())


# z = ql.ZeroCouponBond(2, ql.TARGET(), 100, ql.Date(20,6,2020))
# print(z.bondYield(100, ql.Actual360(), ql.Compounded, ql.Annual))

# fr = ql.FixedRateBond(2, ql.TARGET(), 100.0, ql.Date(15,12,2019), ql.Date(15,12,2024), ql.Period('1Y'), [0.05], ql.ActualActual())
# print(fr.bondYield(100, ql.Actual360(), ql.Compounded, ql.Annual))



# #floating
# schedule = ql.MakeSchedule(ql.Date(15,6,2020), ql.Date(15,6,2022), ql.Period('6m'))
# index = ql.Euribor6M()
# fl = ql.FloatingRateBond(2,100, schedule, index, ql.Actual360(), spreads=[0.01])




# #callable
# schedule = ql.MakeSchedule(ql.Date(15,6,2020), ql.Date(15,6,2022), ql.Period('1Y'))
# putCallSchedule = ql.CallabilitySchedule()

# callability_price  = ql.CallabilityPrice(100, ql.CallabilityPrice.Clean)

# putCallSchedule.append(
#     ql.Callability(callability_price, ql.Callability.Call, ql.Date(15,6,2021))
# )
# callbond =ql.CallableFixedRateBond(2, 100, schedule, [0.01], ql.Actual360(), ql.ModifiedFollowing, 100, ql.Date(15,6,2020), putCallSchedule)
# print(callbond.bondYield(100, ql.Actual360(), ql.Compounded, ql.Annual))


 