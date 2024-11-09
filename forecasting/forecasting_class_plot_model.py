from datetime import datetime
import os
import yaml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from numpy import concatenate
import numpy as np
import pickle
from fire import Fire


def save_loss_plot(history, plot_filename, model_name):
    """Saves the loss plot with a timestamp and includes the model name in the title."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Add model name to the plot title
    plt.title(f'{model_name} Loss over Epochs')

    # Save the plot with a timestamp
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        os.makedirs(output_dir, exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_datetime = f"{plot_filename}_{model_name}_{current_datetime}_loss.png"
        plt.savefig(filename_with_datetime)
        print(f"Loss plot saved as {filename_with_datetime}")

    plt.show()


def evaluate_forecasts(train, actual, predicted, n, y_limits=None, plot_filename=None, model_name=None, save_plots=False):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train) - n, len(train)), train[-n:], label='Train', color='green')
    plt.plot(range(len(train), len(train) + len(actual)), actual, label='Actual', color='blue')
    plt.plot(range(len(train), len(train) + len(predicted)), predicted, label='Predicted', color='red')
    plt.xlabel('Date')
    plt.ylabel(model_name if model_name else 'Model Output')
    plt.legend()
    if y_limits:
        plt.ylim(y_limits)

    if save_plots and plot_filename:
        output_dir = os.path.dirname(plot_filename)
        os.makedirs(output_dir, exist_ok=True)

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_datetime = f"{plot_filename}_{model_name}_{current_datetime}_forecast.png"
        plt.savefig(filename_with_datetime)
        print(f"Plot saved as {filename_with_datetime}")

    plt.show()

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + epsilon))
    return {'MAE': mae, 'RMSE': rmse, 'R-squared': r2, 'MAPE': mape, 'sMAPE': smape}

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def load_data(interval, path):
    load_path = os.path.join(path, interval)
    if not os.path.isdir(load_path):
        raise FileNotFoundError(f"The specified path '{load_path}' does not exist or is not a directory.")

    df_names = []
    loaded_dataframes = {}

    for filename in os.listdir(load_path):
        if filename.endswith('.pkl') and not filename.startswith('.') and os.path.isfile(os.path.join(load_path, filename)):
            df_name = filename[:-4]
            df_names.append(df_name)
            file_path = os.path.join(load_path, filename)
            with open(file_path, 'rb') as file:
                loaded_dataframes[df_name] = pickle.load(file)

    return loaded_dataframes, df_names

from tensorflow.keras.utils import plot_model

def train_lstm(df, config):
    # Extract parameters from config
    prediction_period = config['model']['prediction_period']
    n_in = config['model']['n_in']
    n_out = config['model']['n_out']
    lstm_units = config['model']['lstm_units']
    dropout_rate = config['model']['dropout_rate']
    learning_rate = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    # Data preprocessing
    values = df.values[:-prediction_period].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)

    values = reframed.values
    n_train_hours = values.shape[0] - prediction_period
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X_reshaped = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X_reshaped = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # Model Definition
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential([
        LSTM(lstm_units, input_shape=(train_X_reshaped.shape[1], train_X_reshaped.shape[2])),
        Dropout(dropout_rate),
        Dense(1, activation='relu')
    ])
    model.compile(loss=loss_function, optimizer=optimizer)

    # Plot and save the model architecture
    plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

    # Training the Model
    history = model.fit(train_X_reshaped, train_y, epochs=epochs,
                        batch_size=batch_size, validation_data=(test_X_reshaped, test_y),
                        verbose=2, shuffle=False)

    # Save the loss plot
    save_loss_plot(history, config['evaluation']['loss_plot_filename'], model_name='LSTM')

    # Forecasting and Inverse Scaling
    yhat = model.predict(test_X_reshaped)
    inv_yhat = scaler.inverse_transform(yhat)
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse:.3f}')

    # Plot forecasts
    evaluate_forecasts(df.values, scaler.inverse_transform(test_y), inv_yhat, config['model']['n'],
                       y_limits=config['evaluation']['y_limits'],
                       plot_filename=config['evaluation']['plot_filename'],
                       model_name=config['evaluation']['model_name'],
                       save_plots=config['evaluation']['save_plots'])


def train_lstm_cnn(df, config):
    # Extract parameters from config
    prediction_period = config['model']['prediction_period']
    n_in = config['model']['n_in']
    n_out = config['model']['n_out']
    lstm_units = config['model']['lstm_units']
    dropout_rate = config['model']['dropout_rate']
    cnn_filters = config['model']['cnn_filters']
    cnn_kernel_size = config['model'].get('cnn_kernel_size', 3)
    learning_rate = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']

    # Data Preprocessing
    values = df.values[:-prediction_period].astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_in, n_out)

    values = reframed.values
    n_train_hours = values.shape[0] - prediction_period
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # Reshape input for CNN-LSTM
    train_X_reshaped = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
    test_X_reshaped = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

    # Model Definition: CNN-LSTM
    optimizer = Adam(learning_rate=learning_rate)
    model = Sequential([
        Conv1D(filters=cnn_filters, kernel_size=min(cnn_kernel_size, train_X_reshaped.shape[1]), activation='relu',
               input_shape=(train_X_reshaped.shape[1], 1)),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(1, activation='relu')
    ])
    model.compile(loss=loss_function, optimizer=optimizer)

    # Plot and save the model architecture
    plot_model(model, to_file='lstm_cnn_model.png', show_shapes=True, show_layer_names=True)

    # Training the Model
    history = model.fit(train_X_reshaped, train_y, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_X_reshaped, test_y), verbose=2, shuffle=False)

    # Save the loss plot
    save_loss_plot(history, config['evaluation']['loss_plot_filename'], model_name='CNN_LSTM')

    # Forecasting and Inverse Scaling
    yhat = model.predict(test_X_reshaped)
    inv_yhat = scaler.inverse_transform(yhat)
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse:.3f}')

    # Plot forecasts
    evaluate_forecasts(df.values, scaler.inverse_transform(test_y), inv_yhat, config['model']['n'],
                       plot_filename=config['evaluation']['plot_filename'],
                       model_name=config['evaluation']['model_name'],
                       save_plots=config['evaluation']['save_plots'])

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    interval = config['data']['interval']
    path = config['data']['path']
    i = config['data']['index']

    loaded_dataframes, df_names = load_data(interval, path)

    df = loaded_dataframes[df_names[i]].copy()
    df.set_index(config['data']['index_column'], inplace=True)

    model_type = config['model'].get('model_type', 'lstm')
    if model_type == 'lstm':
        train_lstm(df, config)
    elif model_type == 'lstm_cnn':
        train_lstm_cnn(df, config)
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' specified in config.")

if __name__ == '__main__':
    Fire(main)
