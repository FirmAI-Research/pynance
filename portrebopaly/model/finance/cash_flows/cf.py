import QuantLib as ql 


amount = 105
date = ql.Date(15,6,2020)
cf = ql.SimpleCashFlow(amount, date)
print(cf)

amount = 100
date = ql.Date(15,6,2020)
redemption = ql.Redemption(amount, date)

amount = 100
date = ql.Date(15,6,2020)
ql.AmortizingPayment(amount, date)


amount = 105
nominal = 100.
paymentDate = ql.Date(15,6,2020)
startDate = ql.Date(15,12,2019)
endDate = ql.Date(15,12,2021)

rate = .05
dayCounter = ql.Actual360()
coupon = ql.FixedRateCoupon(endDate, nominal, rate, dayCounter, startDate, endDate)
print(coupon)