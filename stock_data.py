import numpy as np
import math


def scale_range(x, input_range, target_range):

    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * \
        (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range


def train_test_split_linear_regression(stocks):
    # Create numpy arrays for features and targets
    feature = []
    label = []

    # Convert dataframe columns to numpy arrays for scikit learn
    for index, row in stocks.iterrows():
        # print([np.array(row['Item'])])
        feature.append([(row['Item'])])
        label.append([(row['Close'])])

    # Regularize the feature and target arrays and store min/max of input data for rescaling later
    feature_bounds = [min(feature), max(feature)]
    feature_bounds = [feature_bounds[0][0], feature_bounds[1][0]]
    label_bounds = [min(label), max(label)]
    label_bounds = [label_bounds[0][0], label_bounds[1][0]]

    feature_scaled, feature_range = scale_range(
        np.array(feature), input_range=feature_bounds, target_range=[-1.0, 1.0])
    label_scaled, label_range = scale_range(
        np.array(label), input_range=label_bounds, target_range=[-1.0, 1.0])

    # Define Test/Train Split 80/20
    split = .20
    split = int(math.floor(len(stocks['Item']) * split))

    # Set up training and test sets
    X_train = feature_scaled[:-split]
    X_test = feature_scaled[-split:]

    y_train = label_scaled[:-split]
    y_test = label_scaled[-split:]

    return X_train, X_test, y_train, y_test, label_range
    
def train_test_split_ml(stocks, split_size):
    
    stock_len=len(stocks['Item'])
    split = int(math.floor(stock_len * split_size))

    # Setting up training and test sets
    X_train = stocks['Item'][0:split].to_numpy()
    X_test = stocks['Item'][split:stock_len].to_numpy()

    y_train = stocks['Close'][0:split].to_numpy()
    y_test = stocks['Close'][split:stock_len].to_numpy()

    return X_train, X_test, y_train, y_test
    
def convert2matrixDNN(data_arr, look_back):
    X, Y =[], []
    for i in range(len(data_arr)-look_back):
        d=i+look_back  
        X.append(data_arr[i:d,2])
        Y.append(data_arr[d,2])
    return np.array(X), np.array(Y)
    
def train_test_split_dnn(stocks, split_size):

    # setup look_back window 
    look_back = 30
    
    stock_len=len(stocks.values)
    train_size = int(math.floor(stock_len * split_size))
    
    train = stocks.values[0:train_size,:]
    test = stocks.values[train_size:stock_len,:]
    
    trainX, trainY = convert2matrixDNN(train, look_back)
    testX, testY = convert2matrixDNN(test, look_back)
    
    return trainX, trainY, testX, testY
    
def convert2matrixRNN(data_arr, look_back):
   X, Y =[], []
   for i in range(len(data_arr)-look_back):
       d=i+look_back  
       X.append(data_arr[i:d,])
       Y.append(data_arr[d,])
   return np.array(X), np.array(Y)
    
def train_test_split_rnn(stocks, split_size):
    look_back = 30 #create window size as look_back=30
    
    stock_len=len(stocks.values)
    train_size = int(math.floor(stock_len * split_size))
    
    train = stocks.values[0:train_size,2]
    test = stocks.values[train_size:stock_len,2]
    
    test = np.append(test,np.repeat(test[-1,], look_back))
    train = np.append(train,np.repeat(train[-1,],look_back))
    
    trainX,trainY =convert2matrixRNN(train,look_back)
    testX,testY =convert2matrixRNN(test, look_back)
    
    return trainX, trainY, testX, testY


def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut].to_numpy()
    y_train = stocks[prediction_time:-test_data_cut]['Close'].to_numpy()

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].to_numpy()
    y_test = stocks[prediction_time - test_data_cut:]['Close'].to_numpy()

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
