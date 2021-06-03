import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer


def delete_data(data):

    # parameter 'data' containing records of all the stock prices with columns as  ['Date','Open','High','Low','Close','Volume']
    # returns a DataFrame with columns as  ['index','Open','Close','Volume']

    # Define columns to keep as it is
    item = list()
    open = list()
    close = list()
    volume = list()

    # Loop through the stock data and copy the required features
    index = 0
    for ind in range(len(data)):
        item.append(index)
        open.append(data['Open'][ind])
        close.append(data['Close'][ind])
        volume.append(data['Volume'][ind])
        index += 1

    # Create a data frame for stock data
    stocks = pd.DataFrame()

    # Assign fetures to data frame
    stocks['Item'] = item
    stocks['Open'] = open
    stocks['Close'] = pd.to_numeric(close)
    stocks['Volume'] = pd.to_numeric(volume)

    # return the new updated data
    return stocks


def normalize_data(data):

    # Initializing MinMax Scalar
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_StandardScalar(data):

    # Initialize a scaler, then apply it to the features
    scaler = StandardScaler()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_MinMaxScaler(data):

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_MaxAbsScaler(data):

    # Initialize a scaler, then apply it to the features
    scaler = MaxAbsScaler()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_RobustScaler(data):

    # Initialize a scaler, then apply it to the features
    scaler = RobustScaler()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_Normalizer(data):

    # Initialize a scaler, then apply it to the features
    scaler = Normalizer()
    numerical = ['Open', 'Close', 'Volume']
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_QuantileTransformer(data):

    # Initialize a scaler, then apply it to the features
    scaler = QuantileTransformer()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data


def get_normalised_data_PowerTransformer(data):

    # Initialize a scaler, then apply it to the features
    scaler = PowerTransformer()
    numerical = ['Open', 'Close', 'Volume']

    # Applying scalar to features
    data[numerical] = scaler.fit_transform(data[numerical])

    return data
