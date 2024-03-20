from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Feature Engineering: Convert 'ds' to a numeric format (e.g., Unix timestamp)
df_prophet['ds_numeric'] = df_prophet['ds'].astype(np.int64)

# Prepare the features (X) and target (y)
X = df_prophet[['ds_numeric']]
y = df_prophet['y']

# Split the data into training and test sets (using the last 7 points as test for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=7, shuffle=False)

# Initialize and fit the Linear Regression model
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Forecast the next 7 days beyond the last date in the dataset
last_date = df_prophet['ds'].iloc[-1]
future_dates_numeric = [(last_date + pd.Timedelta(days=i)).value for i in range(1, 8)]
future_dates_df = pd.DataFrame(future_dates_numeric, columns=['ds_numeric'])

# Predict the future values
future_preds = model_lr.predict(future_dates_df)

# Prepare the results
future_dates = pd.to_datetime(future_dates_df['ds_numeric'])
forecast_results = pd.DataFrame({'Date': future_dates, 'Predicted Distance Travelled': future_preds})

forecast_results

# Re-import the pandas library and reload the data since the environment was reset
import pandas as pd

# Reload the data
data = pd.read_excel(file_path)

# Convert 'Date and Time' to datetime format
data['Date and Time'] = pd.to_datetime(data['Date and Time'], format='%d-%m-%Y %H:%M')

# Prepare the data for linear regression
data['timestamp'] = data['Date and Time'].astype(np.int64)

# Initialize and fit the Linear Regression model
X = data[['timestamp']]  # Features
y = data['Distance Travelled']  # Target

# We will train on all available data, assuming we want the model to learn from the full dataset
model_lr = LinearRegression()
model_lr.fit(X, y)

# Predict for the next 7 days
last_timestamp = data['timestamp'].iloc[-1]
one_day_in_ns = 24 * 60 * 60 * 1000000000  # nanoseconds in one day
future_timestamps = np.array([last_timestamp + i * one_day_in_ns for i in range(1, 8)]).reshape(-1, 1)

# Predicting the future distances
future_distances = model_lr.predict(future_timestamps)

# Converting future timestamps back to datetime for better readability
future_dates = pd.to_datetime(future_timestamps.flatten())

# Prepare and display the forecast results
forecast_results_linear = pd.DataFrame({
    'Date': future_dates,
    'Predicted Distance Travelled (mm)': future_distances
})
forecast_results_linear


import matplotlib.pyplot as plt

# Plot the historical data
plt.figure(figsize=(10, 6))
plt.plot(data['Date and Time'], data['Distance Travelled'], label='Historical Data')

# Add the forecasted data
plt.plot(forecast_results_linear['Date'], forecast_results_linear['Predicted Distance Travelled (mm)'], label='Forecasted Data', linestyle='--')

plt.title('Distance Travelled Forecast')
plt.xlabel('Date')
plt.ylabel('Distance Travelled (mm)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()



# Plot the historical data with proper scaling
plt.figure(figsize=(12, 7))
plt.plot(data['Date and Time'], data['Distance Travelled'], label='Historical Data', color='blue')

# Since the historical and predicted data have different scales, we convert the forecast dates back for alignment
forecast_dates_aligned = forecast_results_linear['Date']

plt.plot(forecast_dates_aligned, forecast_results_linear['Predicted Distance Travelled (mm)'], label='Forecasted Data', color='red', linestyle='--')

# Highlight the transition from historical to forecasted data
plt.axvline(x=data['Date and Time'].iloc[-1], color='grey', linestyle='--', label='Forecast Start')

plt.title('Distance Travelled Forecast Including Historical Data')
plt.xlabel('Date')
plt.ylabel('Distance Travelled (mm)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import numpy as np

# Preprocess the data for LSTM
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Distance Travelled'].values.reshape(-1,1))

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), 0])
        y.append(data[i + sequence_length, 0])
    return np.array(X), np.array(y)

# Create sequences
sequence_length = 10  # Number of days to use to predict the next value
X, y = create_sequences(scaled_data, sequence_length)

# Reshape X to fit the LSTM input requirement: [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting the dataset into Training and Test sets
split = int(len(X) * 0.9)  # 90% for training, 10% for testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping], shuffle=False)

# Forecasting
# Prepare the last sequence from the training data as the input to forecast the next 7 days
last_sequence = scaled_data[-sequence_length:]
current_sequence = last_sequence.reshape((1, sequence_length, 1))

predicted_distances_scaled = []

# Forecast the next 7 days
for _ in range(7):
    # Get the last prediction, then append it to the sequence to predict the next day
    predicted_distance = model.predict(current_sequence)
    predicted_distances_scaled.append(predicted_distance.flatten()[0])
    
    # Update the sequence to include the new prediction
    current_sequence = np.append(current_sequence[:, 1:, :], [[predicted_distance]], axis=1)

# Inverse transform to get back to the original scale of distances
predicted_distances = scaler.inverse_transform(np.array(predicted_distances_scaled).reshape(-1, 1)).flatten()

# Prepare the dates for the forecasted values
forecast_dates_lstm = pd.date_range(start=data['Date and Time'].iloc[-1] + pd.Timedelta(days=1), periods=7, freq='D')

# Prepare and display the forecast results
forecast_results_lstm = pd.DataFrame({
    'Date': forecast_dates_lstm,
    'Predicted Distance Travelled (mm)': predicted_distances
})
forecast_results_lstm


from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Test for stationarity
def test_stationarity(timeseries):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# Using the 'Distance Travelled' for stationarity test
test_stationarity(data['Distance Travelled'])


# Since the data is not stationary, we'll start with a simple ARIMA model and adjust as necessary
# Initial ARIMA model parameters: p=1, d=1, q=1

model_arima = ARIMA(data['Distance Travelled'], order=(1,1,1))
model_arima_fit = model_arima.fit()

# Forecast the next 7 days
forecast_arima = model_arima_fit.forecast(steps=7)
forecast_dates_arima = pd.date_range(start=data['Date and Time'].iloc[-1], periods=7, freq='D')

# Prepare and display the forecast results
forecast_results_arima = pd.DataFrame({
    'Date': forecast_dates_arima,
    'Predicted Distance Travelled (mm)': forecast_arima
})
forecast_results_arima


# Plot the historical data along with the ARIMA forecast
plt.figure(figsize=(12, 7))
plt.plot(data['Date and Time'], data['Distance Travelled'], label='Historical Data', color='blue')
plt.plot(forecast_results_arima['Date'], forecast_results_arima['Predicted Distance Travelled (mm)'], label='ARIMA Forecast', color='red', linestyle='--')

# Highlight the transition from historical to forecasted data
plt.axvline(x=data['Date and Time'].iloc[-1], color='grey', linestyle='--', label='Forecast Start')

plt.title('Distance Travelled Forecast Including Historical Data (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Distance Travelled (mm)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

