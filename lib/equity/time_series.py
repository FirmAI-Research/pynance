''' Forecast; Technicals
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from talib import RSI, BBANDS, MACD
import yfinance as yf

class Technicals:

    def __init__(self):
        pass

    def ta_dashboard(self, fun_obj):

        fun = fun_obj

        data = yf.download(f"{fun.ticker[0].upper()}", start="2020-01-01", end="2022-10-30")['Adj Close'].to_frame()

        print(data.iloc[-1])

        up, mid, low = BBANDS(data['Adj Close'], timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)

        rsi = RSI(data['Adj Close'], timeperiod=14)

        macd, macdsignal, macdhist = MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        data = pd.DataFrame({'Price': data['Adj Close'], 'BB Up': up, 'BB Mid': mid, 'BB down': low, 'RSI': rsi, 'MACD': macd})

        fig, axes= plt.subplots(nrows=3, figsize=(15, 10), sharex=True)
        data.drop(['RSI', 'MACD'], axis=1).plot(ax=axes[0], lw=1, title='Bollinger Bands')
        data['RSI'].plot(ax=axes[1], lw=1, title='Relative Strength Index')
        axes[1].axhline(70, lw=1, ls='--', c='k')
        axes[1].axhline(30, lw=1, ls='--', c='k')
        data.MACD.plot(ax=axes[2], lw=1, title='Moving Average Convergence/Divergence', rot=0)
        axes[2].set_xlabel('')
        fig.tight_layout()
        sns.despine()