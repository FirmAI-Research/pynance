import QuantLib as ql
import pandas as pd 


fra = ql.ForwardRateAgreement(
    ql.Date(15,6,2020),
    ql.Date(15,12,2020),
    ql.Position.Long,
    0.01,
    1e6,
    ql.Euribor6M(yts),
    yts # yield term structure
)

print(fra)