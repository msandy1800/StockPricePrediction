from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


def build_basic_model(input_dim, output_dim, return_sequences):
    model = Sequential()
    model.add(LSTM(
        units=output_dim,
        input_shape=(None, input_dim),
        return_sequences=return_sequences))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


def build_improved_model(input_dim, output_dim, return_sequences):
    model = Sequential()
    model.add(LSTM(
        units=output_dim,
        input_shape=(None, input_dim),
        return_sequences=return_sequences))

    model.add(Dropout(0.2))

    model.add(LSTM(
        128,
        return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model
