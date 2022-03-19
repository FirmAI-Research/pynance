#
# Python Module with Class
# for Vectorized Backtesting
# of SMA-based Strategies
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
import numpy as np
import pandas as pd
from scipy.optimize import brute
import matplotlib.pyplot as plt 

plt.style.use('seaborn')
plt.rcParams['savefig.dpi'] =300
plt.rcParams['font.family'] = 'serif'


class SMAVectorBacktester(object):
    ''' Class for the vectorized backtesting of SMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        RIC symbol with which to work with
    SMA1: int
        time window in days for shorter SMA
    SMA2: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval

    Methods
    =======
    get_data:
        retrieves and prepares the base data set
    set_parameters:
        sets one or two new SMA parameters
    run_strategy:
        runs the backtest for the SMA-based strategy
    plot_results:
        plots the performance of the strategy compared to the symbol
    update_and_run:
        updates SMA parameters and returns the (negative) absolute performance
    optimize_parameters:
        implements a brute force optimizeation for the two SMA parameters
    '''

    def __init__(self, symbol, SMA1, SMA2, start, end):
        self.symbol = symbol
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.start = start
        self.end = end
        self.results = None
        self.get_data()

    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = pd.read_csv('http://hilpisch.com/pyalgo_eikon_eod_data.csv',
                          index_col=0, parse_dates=True).dropna()
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw / raw.shift(1))
        raw['SMA1'] = raw['price'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['price'].rolling(self.SMA2).mean()
        self.data = raw

    def set_parameters(self, SMA1=None, SMA2=None):
        ''' Updates SMA parameters and resp. time series.
        '''
        if SMA1 is not None:
            self.SMA1 = SMA1
            self.data['SMA1'] = self.data['price'].rolling(
                self.SMA1).mean()
        if SMA2 is not None:
            self.SMA2 = SMA2
            self.data['SMA2'] = self.data['price'].rolling(self.SMA2).mean()

    def run_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace=True)
        data['creturns'] = data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        # gross performance of the strategy
        aperf = data['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
        operf = aperf - data['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)
    
    # @ todo
    def calc_risk_return_stats(self):
        results = self.results.copy().dropna()
        # results[['return', 'strategy']].mean() * 252 # mean returns
        # np.exp(results[['return', 'strategy']].mean() * 252) -1 # log
        # results[['return', 'strategy']].std() * 252 ** 0.5
        # (results[['return', 'strategy']].apply(np.exp) -1).std() * 252 ** 0.5

        results['cumret'] = results['strategy'].cumsum().apply(np.exp)
        results['cummax'] = results['cumret'].cummax()
        results[['cumret','cummax']].dropna().plot(figsize=(10,6))
        plt.show()

        drawdown = results['cummax'] - results['cumret']
        print(drawdown.max())

        temp = drawdown[drawdown ==0]
        periods = (temp.index[1:].to_pydatetime() - temp.index[:-1].to_pydatetime())
        print(periods.max())




    def update_and_run(self, SMA):
        ''' Updates SMA parameters and returns negative absolute performance
        (for minimazation algorithm).

        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.run_strategy()[0]

    def optimize_parameters(self, SMA1_range, SMA2_range):
        ''' Finds global maximum given the SMA parameter ranges.

        Parameters
        ==========
        SMA1_range, SMA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, (SMA1_range, SMA2_range), finish=None)
        return opt, -self.update_and_run(opt)

    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | SMA1=%d, SMA2=%d' % (self.symbol,
                                               self.SMA1, self.SMA2)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))
        plt.show()

    def basic_visualize(self):
        data = self.data.copy().dropna()

        data[['SMA1','SMA2','price']].plot(title='Strategy {}', figsize=(10,6))
        plt.show()


        data['position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['position'].plot(ylim=[-1.1,1.1], title='market Positioning', figsize=(10,6))
        plt.show()

        data['return'] = np.log(data['price'] / data['price'].shift(1))
        data['return'].hist(bins=35, figsize=(10,6))
        plt.show()

        




if __name__ == '__main__':
    smabt = SMAVectorBacktester('EUR=', 100, 200,
                                '2010-1-1', '2020-12-31')
    print(smabt.run_strategy())

    smabt.set_parameters(SMA1=55, SMA2=135)
    print(smabt.run_strategy())


    print(smabt.optimize_parameters((100, 150, 5), (200, 250, 5)))
    smabt.plot_results()
    smabt.basic_visualize()
    smabt.calc_risk_return_stats()
