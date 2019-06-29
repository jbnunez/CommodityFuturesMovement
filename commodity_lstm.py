#lstm.py
#borrowed heavily from 
#https://github.com/BenjiKCF/Neural-Network-with-Financial-Time-Series-Data/blob/master/LSTM_Stock_prediction_20170508.ipynb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import pandas as pd
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, Flatten
from keras.layers.recurrent import LSTM
from keras.models import load_model, Model
import keras
import quandl
import cornsoydata as csd
import classifiers as clf
quandl.ApiConfig.api_key = "63gdVnc_-LzW9XyB1Ajk"




#These are all presently set to their best from the github page's testing
#I added the code to optimize the neurons and dropout parameters.

def build_mlp_model(ishape, neurons, d, decay):
    act_func = 'sigmoid'
    input_layer = Input(shape=ishape)
    hidden = []
    for i in range(len(neurons)):
        if i == 0:
            prev_layer = input_layer
        else:
            prev_layer = hidden[i-1]

        hidden.append(Dense(neurons[i], activation='linear', use_bias=False, 
        kernel_initializer='lecun_normal', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None)(prev_layer))

    #flatten_layer = Flatten()(hidden[len(hidden)-1])

    #output layer simply takes a weighted sum of the outputs 
    #of the last hidden layer
    output_layer = Dense(1, activation='sigmoid', use_bias=False, 
        kernel_initializer='lecun_normal', bias_initializer='zeros', 
        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, bias_constraint=None)(hidden[-1])

    model = Model(inputs=[input_layer], outputs=[output_layer])
    adam = keras.optimizers.Adam(decay=decay)
    model.compile(loss='mse',optimizer='adam')
    model.summary()
    return model




def build_lstm_model(shape, neurons, d, decay):
    model = Sequential()

    model.add(LSTM(neurons[0], input_shape=(shape[0], shape[-1]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=shape[1:], return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='sigmoid'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=decay)
    model.compile(loss='mse',optimizer='adam')
    model.summary()
    return model




#function to evaluate a model's performance while optimizing hyperparameters
def model_score(model, X_train, y_train, X_test, y_test):
    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print("Train Score:", trainScore)
    #print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print("Test Score:", testScore)
    #print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))
    return trainScore, testScore




#function to calculate percentage difference between target and prediction
def percentage_difference(model, X_test, y_test):
    percentage_diff=[]

    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u

        percentage_diff.append(((pr-y_test[u])/pr)*100)
    return p


def test_mlp_model(d, shape, neurons, decay, epochs, X_train, y_train, X_test, y_test, name):
    model = build_mlp_model(shape, neurons, d, decay)
    bsize = len(y_train)//20
    model.fit(X_train, y_train, batch_size=bsize, epochs=epochs, validation_split=0.2, verbose=1)
    # model.save('LSTM_Stock_prediction-20170429.h5')
    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)
    
    X = np.concatenate((X_test, X_train))
    y = np.concatenate((y_test, y_train))
    timesteps = np.linspace(0, len(y_train)+len(y_test), len(y_train)+len(y_test))
    y_pred = model.predict(X)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Time', fontsize = 15)
    ax.set_ylabel('Price', fontsize = 15)
    ax.set_title(name, fontsize = 20)
    #print(y_pred)
    actual = plt.plot(timesteps, y, label='Actual Price')
    predicted = plt.plot(timesteps, y_pred, label='Predicted Price')
    plt.legend()
    plt.savefig(name+".png")
    #plt.show()
    plt.close()

    y_test_pred = model.predict(X)
    m = len(y)-1
    count=0
    up=0
    for i in range(1,m):
        if y[i]>=y[i-1]:
            up+=1
        if y_test_pred[i]>=y[i-1] and y[i]>=y[i-1]:
            count+=1
        elif y_test_pred[i]<y[i-1] and y[i]<y[i-1]:
            count+=1
    percent = float(count)/m
    return trainScore, testScore, percent, float(up)/m


#builds and tests a model
def test_lstm_model(d, shape, neurons, decay, epochs, X_train, y_train, X_test, y_test, name):
    #print(shape)
    model = build_lstm_model(shape, neurons, d, decay)
    bsize = len(y_train)//20
    model.fit(X_train, y_train, batch_size=bsize, epochs=epochs, validation_split=0.2, verbose=1)
    # model.save('LSTM_Stock_prediction-20170429.h5')
    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)

    X = np.concatenate((X_test, X_train))
    y = np.concatenate((y_test, y_train))
    timesteps = np.linspace(0, len(y_train)+len(y_test), len(y_train)+len(y_test))
    y_pred = model.predict(X)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Time', fontsize = 15)
    ax.set_ylabel('Price', fontsize = 15)
    ax.set_title(name, fontsize = 20)

    actual = plt.plot(timesteps, y, label='Actual Price')
    predicted = plt.plot(timesteps, y_pred, label='Predicted Price')
    plt.legend()
    plt.savefig(name+".png")
    #plt.show()
    plt.close()
    y_test_pred = model.predict(X)
    m = len(y_test)-1
    count=0
    up=0
    for i in range(1,m):
        if y[i]>=y[i-1]:
            up+=1
        if y_test_pred[i]>=y[i-1] and y[i]>=y[i-1]:
            count+=1
        elif y_test_pred[i]<y[i-1] and y[i]<y[i-1]:
            count+=1
    percent = float(count)/m

    return trainScore, testScore, percent, float(up)/m




def main():
    labeltype = 'real'
    #labeltype = 'label'
    seq_len = 22

    (fullcorn, fullsoy, cornlabels, soylabels, cornfeat, cornlab, 
    soyfeat, soylab) = clf.prep_data(labels='real',days_back=seq_len)

    xCornWasde = np.array(fullcorn)
    yCornWasde = np.array(cornlabels)
    xSoyWasde = np.array(fullsoy)
    ySoyWasde = np.array(soylabels)
    
    xCornMkt = np.array(cornfeat)
    yCornMkt = np.array(cornlab)
    xSoyMkt = np.array(soyfeat)
    ySoyMkt = np.array(soylab)

    wlen, wcorndim = xCornWasde.shape
    wlen, wsoydim = xSoyWasde.shape
    #print(xCornWasde.shape)
    #print(xSoyWasde.shape)
    #raise ValueError
    mlen, mdays, mdim = xCornMkt.shape
    wbreak = wlen//5
    xCornWasde_test = xCornWasde[:wbreak]
    yCornWasde_test = yCornWasde[:wbreak]

    xCornWasde_train = xCornWasde[wbreak:]
    yCornWasde_train = yCornWasde[wbreak:]

    mbreak = mlen//5
    xCornMkt_test = xCornMkt[:mbreak]
    yCornMkt_test = yCornMkt[:mbreak]

    xCornMkt_train = xCornMkt[mbreak:]
    yCornMkt_train = yCornMkt[mbreak:]
    #print(xCornMkt_train[0])


    xSoyWasde_test = xSoyWasde[:wbreak]
    ySoyWasde_test = ySoyWasde[:wbreak]

    xSoyWasde_train = xSoyWasde[wbreak:]
    ySoyWasde_train = ySoyWasde[wbreak:]

    xSoyMkt_test = xSoyMkt[:mbreak]
    ySoyMkt_test = ySoyMkt[:mbreak]

    xSoyMkt_train = xSoyMkt[mbreak:]
    ySoyMkt_train = ySoyMkt[mbreak:]
    
    #print(xCornMkt[0].shape)
    d = 0.3
    wcshape = (wcorndim,) # feature, window, output
    wsshape = (wsoydim,)
    mshape = (mdays, mdim)
    mneurons = [256, 256, 32, 1]
    wneurons = [256, 256, 32]
    epochs = 20
    decay = 0.4

    

    cornwasde = test_mlp_model(d, wcshape, wneurons, decay, epochs, 
        xCornWasde_train, yCornWasde_train, xCornWasde_test, yCornWasde_test, 'Corn WASDE Predictions')
    print("Corn Wasde:", cornwasde)

    cornmkt = test_lstm_model(d, mshape, mneurons, decay, epochs, 
        xCornMkt_train, yCornMkt_train, xCornMkt_test, yCornMkt_test, 'Corn Market Predictions')
    print("Corn Market:", cornmkt)

    soywasde = test_mlp_model(d, wsshape, wneurons, decay, epochs, 
        xSoyWasde_train, ySoyWasde_train, xSoyWasde_test, ySoyWasde_test, 'Soy WASDE Predictions')
    print("Soy Wasde:", soywasde)

    soymkt = test_lstm_model(d, mshape, mneurons, decay, epochs, 
        xSoyMkt_train, ySoyMkt_train, xSoyMkt_test, ySoyMkt_test, 'Soy Market Predictions')
    print("Soy Market:", soymkt)





    

main()