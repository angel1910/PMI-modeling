
##Syntax yang digunakan dalam penelitian
#Multivariate Long Short-Term Memory (LSTM)
import scipy
import numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
from IPython.core.pylabtools import figsize
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns

from pandas import read_csv 
from pandas import DataFrame
from tscv import GapWalkForward
from keras.wrappers.scikit_learn import KerasRegressor

# Set random seed 
np.random.seed(7) 

# Load data from Excel file
file_path = 'D:\Skripsi\data.xlsx'
df = pd.read_excel(file_path)
# Explore the first five rows
print(df.head())
# Data description
print(df.describe())
print(df.dtypes)


# Plot the three variables in one graph
plt.figure(figsize=(12, 6))
plt.plot(df['Jumlah Penempatan'], label='Jumlah Penempatan')
plt.plot(df['Jumlah Pengaduan'], label='Jumlah Pengaduan')
plt.plot(df['Inflasi'], label='Inflasi')
# Add labels and legend
plt.xlabel('Bulan')
plt.ylabel('Nilai')
plt.title('Plot of Jumlah Penempatan, Jumlah Pengaduan, and Inflasi')
plt.legend()
# Show the plot
plt.show()


# Split the dataset into train and test data
train_size = int(len(df) * 0.82)
train_dataset, test_dataset = df.iloc[:train_size], df.iloc[train_size:]

# Split train data to X and y
X_train_numeric = train_dataset.drop(['Date'], axis=1)
y_train = train_dataset.loc[:, ['Jumlah Penempatan']]
# Split test data to X and y
X_test_numeric = test_dataset.drop(['Date'], axis=1)
y_test = test_dataset.loc[:, ['Jumlah Penempatan']]


#Data transformation
# Different scaler for input and output
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler using available training data
input_scaler = scaler_x.fit(X_train_numeric)  # X_train_numeric should contain only numeric features
output_scaler = scaler_y.fit(y_train)
# Apply the scaler to training data
train_y_norm = output_scaler.transform(y_train)
train_x_norm = input_scaler.transform(X_train_numeric)
# Apply the scaler to test data
test_y_norm = output_scaler.transform(y_test)
test_x_norm = input_scaler.transform(X_test_numeric)  # X_test_numeric should contain only numeric features


# Normalize the data
scaler = MinMaxScaler()
dfforgr = df.drop(['Date'], axis=1)
df_normalized = pd.DataFrame(scaler.fit_transform(dfforgr), columns=dfforgr.columns)
# Plot the normalized data
plt.figure(figsize=(12, 6))
plt.plot(df_normalized['Jumlah Penempatan'], label='Jumlah Penempatan')
plt.plot(df_normalized['Jumlah Pengaduan'], label='Jumlah Pengaduan')
plt.plot(df_normalized['Inflasi'], label='Inflasi')
# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Nilai yang sudah dinormalisasi')
plt.legend()
# Show the plot
plt.show()


#Create a 3D Input Dataset
def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 1
X_test, y_test = create_dataset(test_x_norm, test_y_norm, TIME_STEPS)
X_train, y_train = create_dataset(train_x_norm, train_y_norm, TIME_STEPS)
print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)
print('y_test.shape: ', y_test.shape)

# Create LSTM model
def create_model(units, m):
    model = Sequential()
    # First layer of LSTM
    model.add(m (units = units, return_sequences = True,
                 input_shape = [X_train.shape[1], X_train.shape[2]]))
    model.add(Dropout(0.2))
    # Second layer of LSTM
    model.add(m (units = units))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1))
    #Compile model
    model.compile(loss='mse', optimizer='adam')
    return model

model_lstm = create_model(4, LSTM)

# Fit the models
# Fit LSTM 
def fit_model(model, learning_rate=0.001):  # Add learning_rate as a hyperparameter
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Set the optimizer with the provided learning rate
    model.compile(loss='mean_squared_error', optimizer=optimizer)  # Compile the model with the optimizer
    # shuffle = False because the order of the data matters
    history = model.fit(X_train, y_train, epochs=200, validation_split=0.2,
                        batch_size=32, shuffle=False, callbacks=[early_stop])
    
    # Get the index of the epoch with the lowest validation loss
    best_epoch = np.argmin(history.history['val_loss']) + 1
    # Get the lowest validation loss
    best_val_loss = np.min(history.history['val_loss'])
    # Get the average training loss
    avg_train_loss = np.mean(history.history['loss'])
    
    print("Best Epoch:", best_epoch)
    print("Best Validation Loss:", best_val_loss)
    print("Average Training Loss:", avg_train_loss)
    
    return history

# Tuning hyperparameters with learning rate = 0.001
history_lstm = fit_model(model_lstm, learning_rate=0.001)


#Inverse target variable for train and test data
# Note that I have to use scaler_y
y_test = scaler_y.inverse_transform(y_test)
y_train = scaler_y.inverse_transform(y_train)


#Make prediction using BiLSTM, LSTM and GRU data testing
def prediction(model):
    prediction = model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction

prediction_lstm = prediction(model_lstm)



#Plot true future vs prediction
def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))

    range_future = len(prediction)

    plt.plot(np.arange(range_future), np.array(y_test), label='Data Aktual')
    plt.plot(np.arange(range_future), np.array(prediction),label='Prediksi')

    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Jumlah Penempatan')

plot_future(prediction_lstm, 'LSTM', y_test)


#Calculate RMSE, MAE, dan MAPE

# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = np.mean(np.abs(errors / actual)) * 100  # Calculate MAPE

    print(model_name + ':')
    print('Mean Absolute Error: {:.2f}'.format(mae))
    print('Root Mean Square Error: {:.2f}'.format(rmse))
    print('Mean Absolute Percentage Error: {:.2f}%'.format(mape))
    print('')

# Usage
evaluate_prediction(prediction_lstm, y_test, 'LSTM')


#Make prediction using  LSTM  training
def prediction(model):
    prediction = model.predict(X_train)
    prediction = scaler_y.inverse_transform(prediction)
    return prediction
prediction_lstm = prediction(model_lstm)


#Plot true future vs prediction
def plot_future(prediction, model_name, y_train):
    plt.figure(figsize=(10, 6))

    range_future = len(prediction)

    plt.plot(np.arange(range_future), np.array(y_train), label='Data Aktual')
    plt.plot(np.arange(range_future), np.array(prediction),label='Prediksi')

    plt.title('True future vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Date')
    plt.ylabel('Jumlah Penempatan)')

plot_future(prediction_lstm, 'LSTM', y_train)


#Calculate RMSE, MAE, dan MAPE

# Define a function to calculate MAE and RMSE
def evaluate_prediction(predictions, actual, model_name):
    errors = predictions - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    mape = np.mean(np.abs(errors / actual)) * 100  # Calculate MAPE

    print(model_name + ':')
    print('Mean Absolute Error: {:.2f}'.format(mae))
    print('Root Mean Square Error: {:.2f}'.format(rmse))
    print('Mean Absolute Percentage Error: {:.2f}%'.format(mape))
    print('')

# Usage
evaluate_prediction(prediction_lstm, y_train, 'LSTM')

