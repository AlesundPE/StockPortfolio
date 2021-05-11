import yfinance as yf
import numpy as np
import pandas
import math
import keras
import datetime
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from django.shortcuts import render
from django.http import HttpResponse

def prediction(request, symbol):
    import json

    #Get history data through YFinance
    dateToday = datetime.datetime.now().strftime("%Y-%m-%d")
    date2yrs = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime("%Y-%m-%d")
    rawData = yf.download(symbol, start=date2yrs, end=dateToday)
    data = pandas.DataFrame(rawData)

    #Read and preprocess data
    dataframe = pandas.DataFrame(data)
    dataset = dataframe["Close"]
    dataset = dataset.values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    #Dataset normalizing
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    #Train and test 
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    def create_dataset(dataset, time_step = 1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i : (i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        
        #Data format [samples, timestep]
        dataX = np.array(dataX)
        #Data format [samples, timestep, feature]
        dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
        dataY = np.array(dataY) 	
        return dataX, dataY

    #Datasets
    trainX, trainY = create_dataset(train, 7)
    testX, testY = create_dataset(test, 7)

    #Building Neural network
    LSTM = Sequential()
    LSTM.add(layers.LSTM(32, input_shape=(7, 1)))
    LSTM.add(layers.Dense(1))

    #Loss function & optimizer
    LSTM.compile(loss='mean_squared_error', optimizer='adam')

    #Train
    history_LSTM = LSTM.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=16)


    def predictTargetDays(model, previousDays):
        predict = []
        for i in range(7):
            next_open = model.predict(np.array([previousDays]))
            predict.append(next_open[0])
            # Remove the first one and add the predict value to the end
            previousDays = np.append(previousDays[1:], next_open, axis=0)
        
        nextDays = np.array(predict)
        result = scaler.inverse_transform(nextDays)
        return result


    tempNp = predictTargetDays(LSTM, dataset[-7:])
    tempList = []
    for num in range(len(tempNp)):
        tempList.append(float(str(round(tempNp[num][0],3))))

    tempJson = json.dumps(tempList)

    return render(request, 'prediction.html', {'tickerData': tempJson, "tickerSymbol": symbol})