import pandas as pd 
import numpy as np 


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam

class nnbt():
    '''
    neural network backtest
    '''
    def __init__(self, symbol, start, end, amount, tc, model):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
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
        raw['returns'] = np.log(raw / raw.shift(1))
        self.data = raw.dropna()

    
    def go(self):
        data = self.data.copy()
        data.rename(columns={self.symbol: 'price'}, inplace=True)
        data['return'] = np.log(data['price'] /
                         data['price'].shift(1)) 
        data['direction'] = np.where(data['return'] > 0, 1, 0)
        lags = 5
        cols = []
        for lag in range(1, lags + 1): # <5>
            col = f'lag_{lag}'
            data[col] = data['return'].shift(lag) # <6>
            cols.append(col)
        data.dropna(inplace=True) # <7>
        print(data.round(4).tail())
        print(data)

        optimizer = adam(learning_rate=0.0001)
        def set_seeds(seed=100):
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(100)

        set_seeds()
        model = Sequential()
        model.add(Dense(64, activation='relu',
                input_shape=(lags,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid')) # <5>
        model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        cutoff = '2017-12-31'
        training_data = data[data.index < cutoff].copy()

        mu, std = training_data.mean(), training_data.std()
        training_data_ = (training_data - mu) / std


        test_data = data[data.index >= cutoff].copy()
        test_data_ = (test_data - mu) / std
        model.fit(training_data[cols],
          training_data['direction'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False)

        res = pd.DataFrame(model.history.history)
        res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')
        model.evaluate(training_data_[cols], training_data['direction'])
        pred = model.predict_classes(training_data_[cols])
        print(pred[:30].flatten())


x = nnbt('.SPX', '2010-1-1', '2019-12-31', 10000, 0.001,'nn')
x.go()