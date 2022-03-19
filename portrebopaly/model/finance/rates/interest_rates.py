import QuantLib as ql 

rate = ql.InterestRate(0.05, ql.Actual360(), ql.Compounded, ql.Annual)
print("Rate: ", rate.rate())
print("DayCount: ", rate.dayCounter())
print("DiscountFactor: ", rate.discountFactor(1))
print("DiscountFactor: ", rate.discountFactor(ql.Date(15,6,2020), ql.Date(15,6,2021)))
print("CompoundFactor: ", rate.compoundFactor(ql.Date(15,6,2020), ql.Date(15,6,2021)))
print("EquivalentRate: ", rate.equivalentRate(ql.Actual360(), ql.Compounded, ql.Semiannual, ql.Date(15,6,2020), ql.Date(15,6,2021)))

factor = rate.compoundFactor(ql.Date(15,6,2020), ql.Date(15,6,2021))
print("ImpliedRate: ", rate.impliedRate(factor, ql.Actual360(), ql.Continuous, ql.Annual, ql.Date(15,6,2020), ql.Date(15,6,2021)))