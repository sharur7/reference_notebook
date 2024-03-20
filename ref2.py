# Convert 'Date and Time' to datetime format and set as index
data['Date and Time'] = pd.to_datetime(data['Date and Time'], dayfirst=True)
data.set_index('Date and Time', inplace=True)

# Ensure 'Distance Travelled' is in mm (seems already in mm, but confirming)
data['Distance Travelled'] = data['Distance Travelled'] * 1  # Assuming it's already in mm

# Plot 'Distance Travelled' over time
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Distance Travelled'], label='Distance Travelled')
plt.xlabel('Date and Time')
plt.ylabel('Distance Travelled (mm)')
plt.title('Distance Travelled Over Time')
plt.legend()
plt.grid(True)
plt.show()


from statsmodels.tsa.stattools import adfuller

# Perform Augmented Dickey-Fuller test to check stationarity
adf_test = adfuller(data['Distance Travelled'])

adf_output = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in adf_test[4].items():
    adf_output['Critical Value (%s)' % key] = value

adf_output


# Differencing the series
data_diff = data['Distance Travelled'].diff().dropna()

# Perform Augmented Dickey-Fuller test again to check stationarity of differenced series
adf_test_diff = adfuller(data_diff)

adf_output_diff = pd.Series(adf_test_diff[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in adf_test_diff[4].items():
    adf_output_diff['Critical Value (%s)' % key] = value

adf_output_diff


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF for the differenced series
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(data_diff, ax=ax1, lags=10)
plot_pacf(data_diff, ax=ax2, lags=10)

plt.tight_layout()
plt.show()


from statsmodels.tsa.arima.model import ARIMA

# Define and fit the ARIMA model with parameters (p=1, d=1, q=1)
model = ARIMA(data['Distance Travelled'], order=(1, 1, 1))
model_fit = model.fit()

# Summary of the model
model_fit.summary()


# Forecasting future values
forecast_steps = 20  # Number of steps ahead to forecast, adjust based on need and data frequency
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps+1, closed='right')
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create a DataFrame for the forecast and confidence intervals
forecast_df = pd.DataFrame({
    'Forecast': forecast_mean,
    'Lower CI': forecast_ci.iloc[:, 0],
    'Upper CI': forecast_ci.iloc[:, 1]
}, index=forecast_index)

# Plotting the forecast along with the historical data and maintenance threshold
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Distance Travelled'], label='Historical Data', color='blue')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='pink', alpha=0.3, label='95% Confidence Interval')
plt.axhline(950000, color='green', linestyle='--', label='Maintenance Threshold (950,000 mm)')
plt.xlabel('Date and Time')
plt.ylabel('Distance Travelled (mm)')
plt.title('Forecasted Distance Travelled & Maintenance Threshold')
plt.legend()
plt.grid(True)
plt.show()

forecast_df


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the data
file_path = 'updated_combined_data.xlsx'  # Update the path if necessary
data = pd.read_excel(file_path)

# Preprocess the data
data['Date and Time'] = pd.to_datetime(data['Date and Time'], dayfirst=True)
data.set_index('Date and Time', inplace=True)

# Check for stationarity
adf_test = adfuller(data['Distance Travelled'])
# Apply differencing if needed based on adf_test results

# Fit ARIMA model (example with p=1, d=1, q=1)
model = ARIMA(data['Distance Travelled'], order=(1, 1, 1))
model_fit = model.fit()

# Forecast future values
forecast_steps = 20  # Adjust based on needs
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# Calculate cumulative sum of forecasted values + last known value
last_known_distance = data['Distance Travelled'].iloc[-1]
forecast_cumsum = forecast_mean.cumsum() + last_known_distance

# Identify when the forecasted distance exceeds the maintenance threshold
maintenance_threshold = 950000  # mm
maintenance_due_index = forecast_cumsum[forecast_cumsum >= maintenance_threshold].index[0] if any(forecast_cumsum >= maintenance_threshold) else None
forecast_threshold_reached = forecast_cumsum.loc[maintenance_due_index] if maintenance_due_index else None

print(f"Maintenance due index: {maintenance_due_index}, Distance: {forecast_threshold_reached}")


# Assuming 'forecast_cumsum' and 'data' are correctly set up from previous steps

# Attempting to visualize the forecast alongside historical data
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(data.index, data['Distance Travelled'], label='Historical Data', color='blue')

# Plot forecasted data - simplifying the approach for plotting
if maintenance_due_index:  # Check if forecast extends to or beyond maintenance threshold
    plt.plot(forecast_cumsum.index, forecast_cumsum, label='Forecast', color='red', linestyle='--')
    plt.axvline(maintenance_due_index, color='purple', linestyle='--', label='Maintenance Due')

plt.axhline(950000, color='green', linestyle='--', label='Maintenance Threshold (950,000 mm)')
plt.xlabel('Date and Time')
plt.ylabel('Distance Travelled (mm)')
plt.title('Forecasted Distance Travelled & Maintenance Threshold')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the number of steps to forecast for the next 7 days, based on existing data frequency
# Assuming a consistent time interval between observations in the data
data_frequency_in_days = (data.index[1] - data.index[0]).total_seconds() / (24*3600)
steps_for_7_days = int(round(7 / data_frequency_in_days))

# Re-fit ARIMA model if necessary (using previously determined parameters for simplicity)
model_7_days = ARIMA(data['Distance Travelled'], order=(1, 1, 1))
model_fit_7_days = model_7_days.fit()

# Forecast for the next 7 days
forecast_7_days = model_fit_7_days.get_forecast(steps=steps_for_7_days)
forecast_mean_7_days = forecast_7_days.predicted_mean

# Calculate cumulative sum of the forecasted values + last known value
forecast_cumsum_7_days = forecast_mean_7_days.cumsum() + last_known_distance

# Check if the forecasted distance exceeds the maintenance threshold within the next 7 days
maintenance_due_within_7_days = forecast_cumsum_7_days[forecast_cumsum_7_days >= 950000].any()

maintenance_due_within_7_days


# Recalculate steps for 7 days based on actual data frequency
# Assume daily frequency if not explicitly given, as a fallback
if not data.index.inferred_freq:
    # Assuming data might be daily if frequency isn't clear, as a conservative default
    steps_for_7_days = 7
else:
    # Use the actual frequency of the data to calculate steps for 7 days
    steps_for_7_days = pd.date_range(data.index[-1], periods=8, freq=data.index.inferred_freq).shape[0] - 1

# Generate accurate forecast dates for the next 7 days
accurate_forecast_dates_7_days = pd.date_range(start=data.index[-1], periods=steps_for_7_days+1, freq=data.index.inferred_freq)[1:]

# Forecast again for the next 7 days, using the accurate step count
forecast_7_days_accurate = model_fit_7_days.get_forecast(steps=steps_for_7_days)
forecast_mean_7_days_accurate = forecast_7_days_accurate.predicted_mean

# Calculate cumulative sum of the accurate forecasted values + last known value
forecast_cumsum_7_days_accurate = forecast_mean_7_days_accurate.cumsum() + last_known_distance

# Plotting with accurate date range for the 7-day forecast
plt.figure(figsize=(14, 7))

# Plot historical data
plt.plot(data.index, data['Distance Travelled'], label='Historical Data', color='blue')

# Plot accurate 7-day forecast
plt.plot(accurate_forecast_dates_7_days, forecast_cumsum_7_days_accurate, label='Accurate 7-Day Forecast', color='red', linestyle='--')

# Maintenance threshold
plt.axhline(950000, color='green', linestyle='--', label='Maintenance Threshold (950,000 mm)')

plt.xlabel('Date and Time')
plt.ylabel('Distance Travelled (mm)')
plt.title('Distance Travelled: Historical Data and Accurate 7-Day Forecast')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Splitting the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fitting the ARIMA model to the train set
model_train = ARIMA(train['Distance Travelled'], order=(1, 1, 1))
model_fit_train = model_train.fit()

# Forecasting the values for the duration of the test set
forecast_test = model_fit_train.get_forecast(steps=len(test))
forecast_test_mean = forecast_test.predicted_mean

# Calculating accuracy metrics
mae = mean_absolute_error(test['Distance Travelled'], forecast_test_mean)
rmse = np.sqrt(mean_squared_error(test['Distance Travelled'], forecast_test_mean))

mae, rmse

# Display unique values in the "Service Status" column to understand its structure
unique_service_status = data['Service Status'].unique()
unique_service_status


# Identify the last date in the dataset
last_date = data.index[-1]

# Calculate the next service date by adding 25 days to the last date
next_service_date = last_date + pd.Timedelta(days=25)

last_date, next_service_date
