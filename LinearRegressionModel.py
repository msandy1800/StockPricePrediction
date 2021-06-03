import numpy as np
import stock_data as sd
from sklearn import linear_model


def price_prediction(lr_model, feature, range):
    x = np.reshape(feature, (feature.shape[0], 1))  # test features
    predicted_price = lr_model.predict(x)
    scaled_prediction, _ = sd.scale_range(
        predicted_price, input_range=[-1.0, 1.0], target_range=range)

    return scaled_prediction.flatten()


def build_model(X, y):
    # instantiating linear regression model
    model = linear_model.LinearRegression()
    X = np.reshape(X, (X.shape[0], 1))  # feature dataset
    y = np.reshape(y, (y.shape[0], 1))  # label dataset
    model.fit(X, y)  # model fitting
    return model
