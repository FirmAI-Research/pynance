# https://github.com/PracticalTimeSeriesAnalysis/BookRepo/blob/master/Ch14/ForecastingStocks.ipynb

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import pdb

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import sklearn
import sklearn.preprocessing

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

import yfinance as yf

def get_equity_data():
    tickers = ['SPY']
    data_source = 'yahoo'
    start_date = '2019-01-01'
    end_date = '2021-12-31'
    return yf.download(tickers, start_date,end_date)


def run():

    df = get_equity_data()
    print(df.shape)

    # predict daily return
    df['Return'] = df.Close - df.Open
    # df.Return.plot()
    # plt.show()
    
    df['DailyVolatility'] = df.High - df.Low
    # df.DailyVolatility.plot()
    # plt.show()

    ewdf = df.ewm(halflife = 10).mean()
    vewdf = df.ewm(halflife = 10).var()
    ((df.DailyVolatility - ewdf.DailyVolatility)/ vewdf.DailyVolatility**0.5 ).plot()
    df['ScaledVolatility'] = ((df.DailyVolatility - ewdf.DailyVolatility)/ vewdf.DailyVolatility**0.5 )
    df['ScaledReturn'] = ((df.Return - ewdf.Return)/ vewdf.Return**0.5 )
    df['ScaledVolume'] = ((df.Volume - ewdf.Volume)/ vewdf.Volume**0.5 )
    df = df.dropna()
    print(df)

    ## now we need to form input arrays and target arrays
    train_df = df[:600] # NOTE: update based on size of data
    test_df = df[600:]
    X = train_df[:(600 - 10)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values
    Y = train_df[10:]["ScaledReturn"].values

    print(X.shape)
    X = np.expand_dims(X, axis = 1)
    print(X.shape)

    # reshape X into 'TNC' form with numpy operations
    X = np.split(X, X.shape[0]/10, axis = 0)
    X = np.concatenate(X, axis = 1)


    # neural network
    NUM_HIDDEN    = 8
    NUM_LAYERS    = 1
    LEARNING_RATE = 1e-2
    EPOCHS        = 10
    BATCH_SIZE    = 64
    WINDOW_SIZE   = 20
    Xinp = tf.placeholder(dtype = tf.float32, shape = [WINDOW_SIZE, None, 3])
    Yinp = tf.placeholder(dtype = tf.float32, shape = [None])

    with tf.variable_scope("scope1", reuse=tf.AUTO_REUSE):
        #rnn_cell = tf.nn.rnn_cell.LSTMCell(NUM_HIDDEN, dtype = tf.float32)
        #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=0.9)
        #rnn_output, states = tf.nn.dynamic_rnn(rnn_cell, Xinp, dtype=tf.float32) 
        
        ## tf.nn.rnn_cell.MultiRNNCell
        cells = [tf.nn.rnn_cell.LSTMCell(num_units=NUM_HIDDEN) for n in range(NUM_LAYERS)]
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        rnn_output, states = tf.nn.dynamic_rnn(stacked_rnn_cell, Xinp, dtype=tf.float32) 
        W = tf.get_variable("W_fc", [NUM_HIDDEN, 1], initializer = tf.random_uniform_initializer(-.2, .2))
        output = tf.squeeze(tf.matmul(rnn_output[-1, :, :], W))
        ## notice we have no bias because we expect average zero return
        loss = tf.nn.l2_loss(output - Yinp)
        opt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        ##opt = tf.train.AdamOptimizer(LEARNING_RATE)
        train_step = opt.minimize(loss)

    sess = tf.Session()
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())


    ## for each epoch
    y_hat_dict = {}
    Y_dict = {}

    in_sample_Y_dict = {}
    in_sample_y_hat_dict = {}

    for ep in range(EPOCHS):
        ## for each offset to create a new series of distinct time series 
        ## (re: overlapping issue we talked about previously)
        epoch_training_loss = 0.0
        for i in range(WINDOW_SIZE):
            X = train_df[:(7000 - WINDOW_SIZE)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values
            Y = train_df[WINDOW_SIZE:]["ScaledReturn"].values

            ## make it divisible by window size
            num_to_unpack = math.floor(X.shape[0] / WINDOW_SIZE)
            start_idx = X.shape[0] - num_to_unpack * WINDOW_SIZE
            X = X[start_idx:] 
            Y = Y[start_idx:]  
            
            X = X[i:-(WINDOW_SIZE-i)]
            Y = Y[i:-(WINDOW_SIZE-i)]                                
            
            X = np.expand_dims(X, axis = 1)
            X = np.split(X, X.shape[0]/WINDOW_SIZE, axis = 0)
            X = np.concatenate(X, axis = 1)
            Y = Y[::WINDOW_SIZE]
            ## TRAINING
            ## now batch it and run a sess
            for j in range(math.ceil(Y.shape[0] / BATCH_SIZE)):
                ll = BATCH_SIZE * j
                ul = BATCH_SIZE * (j + 1)
                
                if ul > X.shape[1]:
                    ul = X.shape[1] - 1
                    ll = X.shape[1]- BATCH_SIZE
                
                training_loss, _, y_hat = sess.run([loss, train_step, output],
                                        feed_dict = {
                                            Xinp: X[:, ll:ul, :], Yinp: Y[ll:ul]
                                        })
                epoch_training_loss += training_loss          
                
                in_sample_Y_dict[ep]     = Y[ll:ul] ## notice this will only net us the last part of data trained on
                in_sample_y_hat_dict[ep] = y_hat
                
            ## TESTING
            X = test_df[:(test_df.shape[0] - WINDOW_SIZE)][["ScaledVolatility", "ScaledReturn", "ScaledVolume"]].values
            Y = test_df[WINDOW_SIZE:]["ScaledReturn"].values
            num_to_unpack = math.floor(X.shape[0] / WINDOW_SIZE)
            start_idx = X.shape[0] - num_to_unpack * WINDOW_SIZE
            X = X[start_idx:] ## better to throw away beginning than end of training period when must delete
            Y = Y[start_idx:]                              
            
            X = np.expand_dims(X, axis = 1)
            X = np.split(X, X.shape[0]/WINDOW_SIZE, axis = 0)
            X = np.concatenate(X, axis = 1)
            Y = Y[::WINDOW_SIZE]
            testing_loss, y_hat = sess.run([loss, output],
                                    feed_dict = { Xinp: X, Yinp: Y })
            ## nb this is not great. we should really have a validation loss apart from testing
            
        print("Epoch: %d   Training loss: %0.2f   Testing loss %0.2f:" % (ep, epoch_training_loss, testing_loss))
        Y_dict[ep] = Y
        y_hat_dict[ep] = y_hat


    print(Y_dict)
    plt.plot(Y_dict[EPOCHS - 1])
    plt.plot(y_hat_dict[EPOCHS - 1], 'r')
    plt.title('Out of sample performance')
    plt.show()


    plt.plot(in_sample_Y_dict[EPOCHS - 1])
    plt.plot(in_sample_y_hat_dict[EPOCHS - 1], 'r')
    plt.title('In sample performance')
    plt.show()

    print(pearsonr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1]))
    print(pearsonr(in_sample_Y_dict[EPOCHS - 1], in_sample_y_hat_dict[EPOCHS - 1]))
    print(spearmanr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1]))
    print(spearmanr(in_sample_Y_dict[EPOCHS - 1], in_sample_y_hat_dict[EPOCHS - 1]))

    plt.plot(Y_dict[EPOCHS - 1])
    plt.plot(y_hat_dict[EPOCHS - 1] * 10, 'r')
    plt.title('Rescaled out of sample performance')
    plt.show()


    plt.plot(in_sample_Y_dict[EPOCHS - 1])
    plt.plot(in_sample_y_hat_dict[EPOCHS - 1] * 10, 'r')
    plt.title('Rescaled in sample performance')
    plt.show()

    plt.plot(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1] * 10, linestyle="", marker="o")
    print(pearsonr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1]))
    print(spearmanr(Y_dict[EPOCHS - 1], y_hat_dict[EPOCHS - 1]))


