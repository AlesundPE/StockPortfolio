import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
import datetime
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler # for normalizing the data
from sklearn.metrics import mean_squared_error

dateToday = datetime.datetime.now().strftime("%Y-%m-%d")
date2yrs = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime("%Y-%m-%d")


data = yf.download("AAPL", start="2019-5-11", end="2021-5-11")

#targetColumns = ["Open", "High", "Low", "Close"]
targetColumns = ["High"]
tempDict = {"Open":[], "High":[], "Low":[], "Close":[]}

for item in targetColumns:
    dataframe = pandas.DataFrame(data)
    dataset = dataframe[item]
    dataset = dataset.values
    dataset = dataset.astype('float32')
    dataset = np.reshape(dataset, (-1, 1))

    scaler = MinMaxScaler(feature_range=(0, 1))
    # normalize the dataset
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step = 1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i : (i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        
        # data format [samples, timestep]
        dataX = np.array(dataX)
        # data format [samples, timestep, feature]
        dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], 1)
        dataY = np.array(dataY) 	
        return dataX, dataY


    time_step = 10
    trainX, trainY = create_dataset(train, time_step)
    testX, testY = create_dataset(test, time_step)

    LSTM = Sequential()
    LSTM.add(layers.LSTM(32, input_shape=(time_step, 1)))
    LSTM.add(layers.Dense(1))

    LSTM.compile(loss='mean_squared_error', optimizer='adam')

    history_LSTM = LSTM.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=16)





    def predict_next_5_days(model, last_5_day):
        predict = []
        for i in range(10):
            next_open = model.predict(np.array([last_5_day]))
            predict.append(next_open[0])
            # remove the first one and add the predict value to the end
            last_5_day = np.append(last_5_day[1:], next_open, axis=0)
        
        next_5day = np.array(predict)
        #plt.plot(scaler.inverse_transform(next_5day))
        return scaler.inverse_transform(next_5day).tolist()
        #plt.show()

    last_5_day = dataset[-10:]
    tempNp = predict_next_5_days(LSTM, last_5_day)
    tempList = []
    for num in range(len(tempNp)):
        tempList.append(round(tempNp[num][0],3))

    tempDict[item].append(tempList)
    print(tempList)

#print(tempDict)
#print(tempDict["High"][0])
#print(tempDict["High"][0])
#print(float(tempDict["High"][0][0][0]))

