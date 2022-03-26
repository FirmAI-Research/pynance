


import nasdaqdatalink
from vendors.nasdaq import Nasdaq
ndq = Nasdaq()
df = ndq.get(name = ndq.core_us_fundamentals)
print(df)