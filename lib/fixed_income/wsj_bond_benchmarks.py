import pandas as pd
import requests


class BondBenchmarks:

    def __init__(self):        
        pass

    def get(self):
        url = 'https://www.wsj.com/market-data/bonds/benchmarks'

        header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
        }

        r = requests.get(url, headers=header)

        dfs = pd.read_html(r.text)

        self.data = dfs[0]

        self.data.columns = ['benchmark', 'close', 'pct_change', 'ytd_total_ret', '52-Wk pct_change', 'yield', 'yield_low', 'yield_high','spread', 'spread_low','spread_high']

        return self


    
    def parse(self):

        df = self.data[self.data['benchmark'].isin(['U.S. Government/Credit', 'U.S. Aggregate', 'U.S. Corporate', 'Intermediate', 'Long-term', 'Double-A-rated (AA)', 'Triple-B-rated (Baa)',
            'High Yield Constrained*', 'Triple-C-rated (CCC)', 'High Yield 100', 'U.S. Agency','Mortgage-Backed', 'Corporate Master', 'High Yield', 'Muni Master','EMU§','France','Germany','Japan','U.K.','Emerging Markets**'
        ])]

        self.df = df



