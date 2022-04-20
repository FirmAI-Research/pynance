import sys,os
lib_dirp = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__))) )) # NOTE: relative path from manage.py to lib/
sys.path.append(lib_dirp) 

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: learn recursive neural net                                                                                                                   │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """


# import lib.learn.rnn.rnn as rnn
# rnn.init()

# import lib.learn.rnn.rnn as rnn


""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: Time series analysis class in lib/                                                                                              │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from lib.time_series.timeseries import TimeSeries
# fp = r"C:\dev\pynance\_tmp\ff\ff_factors.csv"
# ts = TimeSeries(data = fp)
# ts.decomposition()

""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ @Test: regression                                                                                          │
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import pandas as pd
# import numpy as np
# import seaborn as sns
from lib.learn.regression.regression import Regression
fp = r"C:\dev\pynance\_tmp\housing.csv"
df = pd.read_csv(fp)
reg = Regression(data=df, dep_var='price')
reg.binary_encode(var_list =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea'])
reg.cat_encode(var_list = ['furnishingstatus'])
reg.split(test_size=0.3)
reg.scale()
reg.build_model()
print(reg.df)
print(reg.model_summary)
reg.vif()
print(reg.vif)
reg.check_model()
reg.evaluate_fit()
