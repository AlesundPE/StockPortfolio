import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
from keras import layers
from sklearn.preprocessing import MinMaxScaler # for normalizing the data
from sklearn.metrics import mean_squared_error



dataframe = pandas.read_csv('AAPL.csv', usecols=[1, 2, 3, 4], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')


dataset = np.average(dataset, axis=1)
dataset = np.reshape(dataset, (-1, 1))




# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset[:10])



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
 
time_step = 5
trainX, trainY = create_dataset(train, time_step)
testX, testY = create_dataset(test, time_step)


"""# Build RNN and LSTM model

"""

# create and fit the LSTM network
RNN = Sequential()
RNN.add(layers.SimpleRNN(5, input_shape=(time_step, 1))) # (batch, 5)
RNN.add(layers.Dense(1))

LSTM = Sequential()
LSTM.add(layers.LSTM(5, input_shape=(time_step, 1)))
LSTM.add(layers.Dense(1))

"""# Preparing for Training
Choose the suitable loss function, optimizer and metrics
"""

RNN.compile(loss='mean_squared_error', optimizer='adam')
LSTM.compile(loss='mean_squared_error', optimizer='adam')

"""# Training Neural Networks"""

history_RNN = RNN.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=16)

history_LSTM = LSTM.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=16)

"""# visulize the training history"""

plt.plot(history_RNN.history['val_loss'])
plt.plot(history_RNN.history['loss'])
plt.title('RNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history_LSTM.history['val_loss'])
plt.plot(history_LSTM.history['loss'])
plt.title('LSTM model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

"""# Check model performance
In order to check model performance, we need to invert the prediction before calculating error scores to ensure that performance is reported in the same units as the original data.

"""

def check_model(model, model_name, trainX, trainY, testX, testY, time_step):
  print("---------------check {} performance----------------".format(model_name))
  # make predictions
  trainPredict = model.predict(trainX)
  testPredict = model.predict(testX)

  # invert predictions
  trainPredict = scaler.inverse_transform(trainPredict)
  orginal_trainY = scaler.inverse_transform([trainY])
  testPredict = scaler.inverse_transform(testPredict)
  orginal_testY = scaler.inverse_transform([testY])

  # calculate root mean squared error
  trainScore = math.sqrt(mean_squared_error(orginal_trainY[0], trainPredict[:,0]))
  print('Train Score: %.2f RMSE' % (trainScore))
  testScore = math.sqrt(mean_squared_error(orginal_testY[0], testPredict[:,0]))
  print('Test Score: %.2f RMSE' % (testScore))

  # shift train predictions for plotting
  trainPredictPlot = np.empty_like(dataset)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[time_step:len(trainPredict)+time_step, :] = trainPredict

  # shift test predictions for plotting
  testPredictPlot = np.empty_like(dataset)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(trainPredict)+(time_step*2)+1:len(dataset)-1, :] = testPredict

  # plot baseline and predictions
  plt.plot(scaler.inverse_transform(dataset))
  plt.plot(trainPredictPlot)
  plt.plot(testPredictPlot)
  plt.show()



check_model(RNN, "RNN", trainX, trainY, testX, testY, time_step)



check_model(LSTM, "LSTM", trainX, trainY, testX, testY, time_step)



def predict_next_5_days(model, last_5_day):

  predict = []
  for i in range(5):
    next_open = model.predict(np.array([last_5_day]))
    predict.append(next_open[0])
    # remove the first one and add the predict value to the end
    last_5_day = np.append(last_5_day[1:], next_open, axis=0)
  
  next_5day = np.array(predict)
  plt.plot(scaler.inverse_transform(next_5day))
  plt.show()

"""## Predicting with RNN"""

last_5_day = dataset[-5:]
predict_next_5_days(RNN, last_5_day)

"""## Predicting with LSTM"""

predict_next_5_days(LSTM, last_5_day)

"""**Now is 2/15/2021. let's see next 5 days' price, 2/16 - 2/22, excluding weekend.**
 
LSTM show better performance on test dataset. But its next 5 days' prediction looks very interesting.
"""