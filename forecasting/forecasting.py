import os
import sys
import pickle
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Custom functions
# from forecasting_functions import evaluate_forecasts
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt


def evaluate_forecasts(train, actual, predicted, n, y_limits=None, plot_filename=None, model_name=None):
    """
    Evaluates and plots the forecasts for a given model.

    Parameters:
    - train (list or array-like): Training data points.
    - actual (list or array-like): Actual values.
    - predicted (list or array-like): Predicted values.
    - n (int): Number of last data points from the training set to plot.
    - y_limits (tuple, optional): y-axis limits for the plot in the form (min, max).
    - plot_filename (str, optional): Base filename to save the plot (if specified).
    - model_name (str, optional): Name of the model for labeling purposes.

    Returns:
    - Evaluation metric (from evaluate_forecast function): Evaluation metric calculated from the actual and predicted values.
    """

    # Calculate the minimum length for comparison between actual and predicted
    m = min(len(actual), len(predicted))

    # Define plot label names
    x_label = 'Date'  # Modify if X-axis represents something other than date
    y_label = model_name if model_name else 'Model Output'

    # Plot the last 'n' points of the training data in green
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train) - n, len(train)), train[-n:], label='Train', color='green')

    # Plot actual values in blue
    plt.plot(range(len(train), len(train) + m), actual[:m], label='Actual', color='blue')

    # Plot predicted values in red
    plt.plot(range(len(train), len(train) + m), predicted[:m], label='Predicted', color='red')

    # Add labels, legend, and optional y-axis limits
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if y_limits:
        plt.ylim(y_limits[0], y_limits[1])

    # Save the plot if a filename is provided
    if plot_filename:
        plt.savefig(f"{plot_filename}_{model_name}_forecast.png")
        print(f"Plot saved as {plot_filename}_{model_name}_forecast.png")

    # Show the plot
    plt.show()

    # Evaluate and return forecast accuracy
    return evaluate_forecast(actual[:m], predicted[:m])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_forecast(y_true, y_pred):
    """
    Evaluate forecast predictions against true values using multiple metrics.

    Args:
        y_true (numpy.ndarray or list): True values.
        y_pred (numpy.ndarray or list): Predicted values.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Calculate MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)

    # Calculate RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Calculate R-squared (R²) score
    r2 = r2_score(y_true, y_pred)

    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

    # Create a dictionary to store the evaluation metrics
    evaluation_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'MAPE': mape,
        'sMAPE': smape
    }

    return evaluation_metrics

import pandas as pd
from pandas import DataFrame, concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Converts a time series dataset into a supervised learning format.

    Args:
        data (list or array-like): Sequence of observations.
        n_in (int): Number of lag observations (input sequences) to use as input features.
        n_out (int): Number of future observations (output sequences) to use as target values.
        dropnan (bool): Whether to drop rows with NaN values.

    Returns:
        pd.DataFrame: DataFrame formatted for supervised learning with columns representing time-lagged inputs and forecasted outputs.
    """
    # Determine the number of variables in the dataset
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

    # Aggregate all columns
    agg = concat(cols, axis=1)
    agg.columns = names

    # Drop rows with NaN values if specified
    if dropnan:
        agg.dropna(inplace=True)

    return agg
import os
import pickle
import pandas as pd

def load_data(interval, path):
    """
    Loads DataFrames from pickle files within a specified date range and interval.

    Args:
        start_datetime (str): The starting datetime as a string (e.g., '2021-10-05 14:35:14').
        end_datetime (str): The ending datetime as a string (e.g., '2023-08-29 19:57:41').
        interval (str): Time interval folder name (e.g., 'daily', 'hourly').
        path (str): Base directory path where pickle files are located.

    Returns:
        tuple: A dictionary with DataFrame names as keys and loaded DataFrames as values, and a list of DataFrame names.
    """
    # Construct the full path to the specified interval folder
    load_path = os.path.join(path, interval)
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"The specified path '{load_path}' does not exist or is not a directory.")

    df_names = []
    loaded_dataframes = {}

    # List files in the directory and filter out non-pickle files
    for filename in os.listdir(load_path):
        if filename.endswith('.pkl') and not filename.startswith('.') and os.path.isfile(os.path.join(load_path, filename)):
            df_name = filename[:-4]  # Remove '.pkl' from the filename
            df_names.append(df_name)

            # Load each DataFrame from its corresponding pickle file
            file_path = os.path.join(load_path, filename)
            with open(file_path, 'rb') as file:
                loaded_dataframes[df_name] = pickle.load(file)

    return loaded_dataframes, df_names

def train(df):
    m = 24 * 14
    # load dataset
    values = df.values[:-m]
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # split into train and test sets
    reframed_df = pd.DataFrame(reframed)
    values = reframed_df.values
    # values = reframed.values
    n_train_hours = values.shape[0] - m
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X_reshaped = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X_reshaped = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # design network

    # Define the learning rate
    learning_rate = 0.00001

    # Create an Adam optimizer with the specified learning rate
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential()
    model.add(LSTM(105, input_shape=(train_X_reshaped.shape[1], train_X_reshaped.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))
    # model.add(Dense(1, activation='tanh'))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X_reshaped, train_y, epochs=10, batch_size=128, \
                        validation_data=(test_X_reshaped, test_y), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # make a prediction
    yhat = model.predict(test_X)
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(yhat)  # inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    results = evaluate_forecasts(df.values, scaler.inverse_transform(test_y), inv_yhat, 2 * len(test), y_limits=None)
    print(results)

interval = 'H'  # intervals = ['D', 'H', 'T', '10S', 'B', 'W', 'M', 'MS', 'Q', 'QS', 'Y', 'YS', '15D', '3H']
i = 6
path = os.getcwd() + '/' + 'dataset' + '/'

loaded_dataframes, df_names = load_data(interval, path)

df = loaded_dataframes[df_names[i]].copy()
df.set_index('date_creation', inplace=True)
train(df)

