import QuantLib as ql

strike = 100.0
maturity = ql.Date(15,6,2025)
option_type = ql.Option.Call

payoff = ql.PlainVanillaPayoff(option_type, strike)
print(payoff)


binaryPayoff = ql.CashOrNothingPayoff(option_type, strike, 1)

europeanExercise = ql.EuropeanExercise(maturity)
europeanOption = ql.VanillaOption(payoff, europeanExercise)


americanExercise = ql.AmericanExercise(ql.Date().todaysDate(), maturity)
americanOption = ql.VanillaOption(payoff, americanExercise)

bermudanExercise = ql.BermudanExercise([ql.Date(15,6,2024), ql.Date(15,6,2025)])
bermudanOption = ql.VanillaOption(payoff, bermudanExercise)

binaryOption = ql.VanillaOption(binaryPayoff, europeanExercise)