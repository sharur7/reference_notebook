from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Preparing the time series data
ts_data = positive_data.set_index('Date and Time')['Time In Ctrl. Vent']

# Train-test split (80% train, 20% test)
split_point = int(len(ts_data) * 0.8)
train, test = ts_data[:split_point], ts_data[split_point:]

# Fit an ARIMA model (using a simple parameter set for demonstration)
model = ARIMA(train, order=(1,1,1))
fitted_model = model.fit()

# Forecast the next 5 points beyond the test set
forecast = fitted_model.forecast(steps=5)

# Calculate forecast error on the test set for demonstration (if there are enough test points)
test_forecast = fitted_model.forecast(steps=len(test))
mse = mean_squared_error(test, test_forecast[:len(test)]) if len(test) > 0 else None

forecast, mse






# Resetting the index to avoid issues with non-unique datetime indices
test_reset = test.reset_index(drop=True)
test_forecast_reset = test_forecast[:len(test)].reset_index(drop=True)

# Recreate the comparison DataFrame with reset indices
comparison_df_fixed = pd.DataFrame({
    'Actual Values': test_reset,
    'Forecasted Values': test_forecast_reset
})

comparison_df_fixed





# Plotting Actual vs Forecasted Values for visual comparison
plt.figure(figsize=(10, 6))
plt.plot(comparison_df_fixed.index, comparison_df_fixed['Actual Values'], marker='o', linestyle='-', label='Actual Values')
plt.plot(comparison_df_fixed.index, comparison_df_fixed['Forecasted Values'], marker='x', linestyle='--', label='Forecasted Values', color='red')
plt.title('Actual vs. Forecasted Values')
plt.xlabel('Index')
plt.ylabel('Time In Ctrl. Vent (seconds)')
plt.legend()
plt.grid(True)
plt.show()



\





# Load the new unseen Excel file
unseen_file_path = '/mnt/data/updated_combined_data_vol_sp_new.xlsx'
unseen_data = pd.read_excel(unseen_file_path)

# Display the first few rows of the dataframe to understand its structure
unseen_data.head()



# Extracting the first five setpoints for "Time In Ctrl. Vent"
first_five_setpoints = unseen_data['Time In Ctrl. Vent'].head(5)

# Convert the forecast series to a DataFrame for easy comparison
forecast_df_april_8 = pd.DataFrame(forecast_april_8, columns=['Forecasted Time In Ctrl. Vent'])

# Reset index for proper alignment
forecast_df_april_8.reset_index(drop=True, inplace=True)
first_five_setpoints_df = pd.DataFrame(first_five_setpoints).reset_index(drop=True)

# Combine the actual and forecasted data for comparison
comparison_actual_forecast = pd.concat([first_five_setpoints_df, forecast_df_april_8], axis=1)
comparison_actual_forecast.columns = ['Actual Time In Ctrl. Vent', 'Forecasted Time In Ctrl. Vent']

comparison_actual_forecast


