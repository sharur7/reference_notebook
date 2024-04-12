from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Prepare the data (exclude the last 5 observations for testing)
train_data = data['Time In Ctrl. Vent'][:-5]
test_data = data['Time In Ctrl. Vent'][-5:]

# Fit the ARIMA model
model = ARIMA(train_data, order=(1, 1, 1))  # ARIMA parameters are chosen generically here; these may need tuning
fitted_model = model.fit()

# Forecast the next 5 values
forecast = fitted_model.get_forecast(steps=5)
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int(alpha=0.05)  # 95% confidence intervals

# Calculate the mean squared error
mse = mean_squared_error(test_data, forecast_values)

forecast_values, test_data, mse




# Standardize indices for the comparison to ensure alignment
forecast_values.index = test_data.index

# Recalculate the error
comparison_df = pd.DataFrame({
    'Forecasted': forecast_values,
    'Actual': test_data,
    'Error': forecast_values - test_data
})

# Plotting the forecasted vs actual values
plt.figure(figsize=(10, 6))
plt.plot(comparison_df.index, comparison_df['Forecasted'], 'r-', label='Forecasted')
plt.plot(comparison_df.index, comparison_df['Actual'], 'b-', label='Actual')
plt.fill_between(comparison_df.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('Forecast vs Actual Comparison')
plt.xlabel('Date and Time')
plt.ylabel('Time In Control Vent')
plt.legend()
plt.grid(True)
plt.show()

# Display the comparison DataFrame
comparison_df



# Simplified plotting without confidence intervals to avoid the previous error
plt.figure(figsize=(10, 6))
plt.plot(comparison_df.index, comparison_df['Forecasted'], 'r-', label='Forecasted')
plt.plot(comparison_df.index, comparison_df['Actual'], 'b-', label='Actual')
plt.title('Forecast vs Actual Comparison')
plt.xlabel('Date and Time')
plt.ylabel('Time In Control Vent')
plt.legend()
plt.grid(True)
plt.show()

# Display the comparison DataFrame
comparison_df




# Load the new data from the newly uploaded Excel file
new_data_path = '/mnt/data/updated_combined_data_vol_sp_new.xlsx'
new_data = pd.read_excel(new_data_path)

# Display the first few rows of the new dataframe to understand its structure
new_data.head()


# Convert 'Date and Time' to datetime for new data
new_data['Date and Time'] = pd.to_datetime(new_data['Date and Time'])

# Set 'Date and Time' as index for new data
new_data.set_index('Date and Time', inplace=True)

# Combine the previous data with the new data
combined_data = pd.concat([data, new_data])

# Drop duplicates, in case the new data overlaps with the old data
combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

# Show summary of the combined data
combined_data.info(), combined_data.tail()


# Prepare the data for retraining (exclude the last 5 observations for testing)
train_data_new = combined_data['Time In Ctrl. Vent'][:-5]
test_data_new = combined_data['Time In Ctrl. Vent'][-5:]

# Fit the ARIMA model on the new training data
new_model = ARIMA(train_data_new, order=(1, 1, 1))  # Same order as before, could be optimized
fitted_new_model = new_model.fit()

# Forecast the next 5 values
new_forecast = fitted_new_model.get_forecast(steps=5)
new_forecast_values = new_forecast.predicted_mean
new_forecast_conf_int = new_forecast.conf_int(alpha=0.05)  # 95% confidence intervals

# Prepare the forecasted values with test data
new_forecast_values.index = test_data_new.index  # Match the indices

# Calculate errors
new_errors = new_forecast_values - test_data_new

# Create comparison DataFrame
new_comparison_df = pd.DataFrame({
    'Forecasted': new_forecast_values,
    'Actual': test_data_new,
    'Error': new_errors
})

new_comparison_df


# Forecast the next 5 future values beyond the available data
future_forecast = fitted_new_model.get_forecast(steps=5)
future_forecast_values = future_forecast.predicted_mean
future_forecast_conf_int = future_forecast.conf_int(alpha=0.05)  # 95% confidence intervals

# Generate a date range for the new forecasted points, assuming consecutive time points
last_date = combined_data.index[-1]
future_dates = pd.date_range(start=last_date, periods=6, freq='T')[1:]  # 'T' stands for minutely frequency

# Assign the date range to the forecasted values
future_forecast_values.index = future_dates

# Display the forecasted values
future_forecast_values


import pandas as pd

# Assume 'data' is your complete dataset and 'forecasted_data' contains the forecast dates
# Load the complete dataset
data_path = 'path_to_your_complete_dataset.xlsx'
data = pd.read_excel(data_path)

# Load the forecasted data (or it could be a DataFrame created during your session)
forecasted_data_path = 'path_to_your_forecasted_data.xlsx'
forecasted_data = pd.read_excel(forecasted_data_path)

# Ensure both dataframes have 'Date and Time' in datetime format and set as index
data['Date and Time'] = pd.to_datetime(data['Date and Time'])
data.set_index('Date and Time', inplace=True)

forecasted_data['Date and Time'] = pd.to_datetime(forecasted_data['Date and Time'])
forecasted_data.set_index('Date and Time', inplace=True)

# Extract actual values that correspond to the forecasted data dates
actual_values = data.loc[forecasted_data.index, 'Time In Ctrl. Vent']

# Print the extracted actual values
print(actual_values)



import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load the complete dataset
data_path = 'path_to_your_complete_dataset.xlsx'
data = pd.read_excel(data_path)

# Load the forecasted data to identify new data points
forecasted_data_path = 'path_to_your_forecasted_data.xlsx'
forecasted_data = pd.read_excel(forecasted_data_path)

# Ensure both dataframes have 'Date and Time' in datetime format and set as index
data['Date and Time'] = pd.to_datetime(data['Date and Time'])
data.set_index('Date and Time', inplace=True)

forecasted_data['Date and Time'] = pd.to_datetime(forecasted_data['Date and Time'])
forecasted_data.set_index('Date and Time', inplace=True)

# Extract actual values that correspond to the forecasted data dates
actual_values = data.loc[forecasted_data.index, 'Time In Ctrl. Vent']

# Assuming 'actual_values' are now part of your forecasted_data or updated data
updated_data = pd.concat([data, actual_values]).drop_duplicates()

# Retrain the model on the updated dataset
# Assuming you've previously determined the order (p, d, q) for the ARIMA model
model = ARIMA(updated_data['Time In Ctrl. Vent'], order=(1, 1, 1))
fitted_model = model.fit()

# Now your model is updated and ready to make new forecasts or be used further
print("Model retrained with the updated data.")

# Optionally, you could forecast future points or perform further analysis
new_forecast = fitted_model.get_forecast(steps=5)
print(new_forecast.predicted_mean)
