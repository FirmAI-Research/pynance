#  python -m pip install QuantLib
import QuantLib as ql


class Pricing:

    def __init__(self, valuation_date=None) -> None:
        self.valuation_date = ql.Date(1, ql.January, 2017)
        ql.Settings.instance().evaluationDate = self.valuation_date

        self.calendar = ql.UnitedStates()

        self.flat_forward()


    def flat_forward(self):
        settlementDays = 2
        forwardRate = 0.05
        dayCounter = ql.Actual360()
        # Construct flat forward rate term structure
        flatForwardTermStructure = ql.FlatForward(settlementDays, self.calendar, forwardRate, dayCounter)
        flatForwardTermStructure.referenceDate()
        print("Max Date: ", flatForwardTermStructure.maxDate())

        effectiveDate = ql.Date(15, 6, 2020)
        terminationDate = ql.Date(15, 6, 2022)
        schedule = ql.MakeSchedule(effectiveDate, terminationDate, ql.Period('6M'))
        notional = [100.0]
        rate = [0.05]
        leg = ql.FixedRateLeg(schedule, dayCounter, notional, rate)
        print(leg)

        dayCounter = ql.Thirty360()
        rate = 0.03
        compoundingType = ql.Compounded
        frequency = ql.Annual
        interestRate = ql.InterestRate(rate, dayCounter, compoundingType, frequency)
        print(interestRate)

        # ql.Settings.instance().evaluationDate = ql.Date(15,12,2020)
        # print( ql.CashFlows.npv(leg, rate) )




