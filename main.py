from flask import Flask , render_template , request , Response , send_file
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense , LSTM

app = Flask(__name__ , static_folder="static")


@app.route('/',methods=['GET'])
def render_html():
    """
        This function renders the HTML page
     """
    return render_template('index.html')


@app.route('/',methods=['POST'])
def stock_model():
    """
        This function load the data, preprocess it, run the neural network model
        and plot the final results
     """
    stock_name = request.form['stock_name']
    data = yf.download(stock_name, start="2017-01-01", end="2021-12-20")
    # new dataframe with only the target feature
    data_close = data.filter(['Adj Close'])
    # convert to numpy array
    data_close_array = data_close.values
    # get training data length
    training_data_length = math.ceil(len(data_close_array) * .8)
    # scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_close_array)
    #create train data split
    train_data = scaled_data[0:training_data_length , :]
    #split x and y
    x_train = []
    y_train = []
    for i in range (60 , len(train_data)):
        x_train.append(train_data[i-60:i , 0])
        y_train.append(train_data[i,0])
    # convert the x train and y train data to numpy array
    x_train, y_train = np.array(x_train), np.array(y_train)
    # reshape the data to fit into the lstm input
    # samples number is the train data length , 60 is the time step ,
    # and we have only one feature which is the adj closing price
    x_train = np.reshape(x_train, (len(x_train), x_train.shape[1], 1))

    # building the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # create the testing dataset
    # create a new array containing scaled values
    test_data = scaled_data[training_data_length - 60:, :]
    # create the datasets x_test , y_test
    x_test = []
    y_test = data_close_array[training_data_length:, :]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # convert the data to numpy array
    x_test = np.array(x_test)

    # reshape the data to 3d
    # samples numbers , time steps , feature count
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # get the predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # get root mean squared error
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)

    # plot the data
    train = data[:training_data_length]
    valid = data[training_data_length:]
    valid['Predictions'] = predictions
    plt.title('Stock Prediction Model')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD($)', fontsize=18)
    plt.plot(train['Adj Close'])
    plt.plot(valid[['Adj Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.savefig('static/images/plot.png')

    return render_template('index.html' , url='/static/images/plot.png' ,rmse=rmse , stock_name=stock_name)

if __name__ == '__main__':
    app.run(port=3000 , debug=True)