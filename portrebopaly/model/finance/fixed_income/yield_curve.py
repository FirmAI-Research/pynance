import QuantLib as ql
import matplotlib.pyplot as plt

dates = [ql.Date(15,6,2020), ql.Date(15,6,2021), ql.Date(15,6,2022)]
zeros = [0.01, 0.02, 0.03]
curve = ql.ZeroCurve(dates, zeros, ql.ActualActual(), ql.TARGET())

curve.nodes()

plt.plot(*list(zip(*[(dt.to_date(), rate) for dt,rate in curve.nodes()])), marker='o')
plt.show()