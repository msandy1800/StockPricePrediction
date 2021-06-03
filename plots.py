import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 10)


def price(x):
    # returns the price upto 2 decimal digits and dollar sign added in the beginning
    return '$%1.2f' % x


def prediction_plot(actual, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel('Price (USD)')  # depicts the y-axis label
    plt.xlabel('Trading (Days)')  # depicts the x-axis label

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Sets title of the plot
    ax.set_title('Actual trading (green) vs prediction (blue)')
    ax.legend(loc='upper left')

    # Prints the plot against stock and its closing value
    plt.show()


def basic_plot(stocks):

    fig, ax = plt.subplots()
    ax.plot(stocks['Item'], stocks['Close'], '#178196')

    ax.format_ydata = price
    ax.set_title('Actual Trading')

    # Add labels
    plt.ylabel('Price (USD)')  # depicts the y-axis label
    plt.xlabel('Trading (Days)')  # depicts the x-axis label

    # Prints the plot against stock and its closing value
    plt.show()


def lstm_prediction_plot(actual, prediction):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel('Price (USD)')  # depicts the y-axis label
    plt.xlabel('Trading (Days)')  # depicts the x-axis label

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Sets title of the plot
    ax.set_title('Actual trading (green) vs prediction (blue)')
    ax.legend(loc='upper left')

    # Prints the plot against stock and its closing value
    plt.show()
