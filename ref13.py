# Convert 'Total Time' and 'Total Time Difference' from milliseconds to seconds
new_data['Total Time (seconds)'] = new_data['Total Time'] / 1000
new_data['Total Time Difference (seconds)'] = new_data['Total Time Difference'] / 1000

# Display the data with the converted times
converted_data = new_data[['Date and Time', 'Set Point Count', 'Total Time (seconds)', 'Total Time Difference (seconds)']]
converted_data.head(10)




from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare the data for modeling
X = new_data[['Set Point Count', 'Total Time']]  # Independent variables
y = new_data['Time In Ctrl. Vent']  # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Assuming a similar pattern for the next 10 setpoints, we increment set point count and estimate total time
# For simplicity, use the last known 'Total Time' as a baseline and assume slight random variations around it
last_set_point = new_data['Set Point Count'].iloc[-1]
last_total_time = new_data['Total Time'].iloc[-1]

# Generate data for next 10 setpoints
new_set_points = np.arange(last_set_point + 1, last_set_point + 11)
new_total_times = last_total_time + np.random.normal(0, 500, 10)  # Random normal variations around the last total time

# New features data frame
new_features = np.vstack((new_set_points, new_total_times)).T

# Predict 'Time In Ctrl. Vent' for the next 10 setpoints
predicted_times = model.predict(new_features)

# Create a DataFrame to display predictions neatly
predictions_df = pd.DataFrame({
    'Set Point Count': new_set_points,
    'Estimated Total Time': new_total_times,
    'Predicted Time In Ctrl. Vent': predicted_times
})
predictions_df








# Checking the most common time differences to estimate intervals (excluding zero seconds differences)
common_intervals = new_data['Time Difference'][new_data['Time Difference'] > 0].mode()

# Get the last known date and time from the dataset
last_known_datetime = new_data['Date and Time'].iloc[-1]

# If there's a common interval, use it; otherwise default to a reasonable guess (e.g., 60 seconds)
estimated_interval = common_intervals.iloc[0] if not common_intervals.empty else 60

# Generate future datetimes based on the estimated interval
future_datetimes = [last_known_datetime + pd.to_timedelta(estimated_interval * i, unit='s') for i in range(1, 11)]

# Add these to the predictions DataFrame
predictions_df['Estimated Date and Time'] = future_datetimes

# Show the updated predictions DataFrame with estimated datetimes
predictions_df






# Redefine the estimated interval as 14 seconds (middle of the 13 to 15 seconds range)
revised_interval = 14  # seconds

# Generate new future datetimes based on the revised interval of 14 seconds
revised_future_datetimes = [last_known_datetime + pd.to_timedelta(revised_interval * i, unit='s') for i in range(1, 11)]

# Update the predictions DataFrame with the new estimated datetimes
predictions_df['Revised Estimated Date and Time'] = revised_future_datetimes

# Display the updated predictions DataFrame with the new estimated datetimes
predictions_df










from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Prepare the time series data
time_series = new_data['Time In Ctrl. Vent']

# Perform a Dickey-Fuller test to check stationarity
result = adfuller(time_series.dropna())  # dropna() handles any NaN values in the series
adf_statistic = result[0]
p_value = result[1]

# Display the results of the Dickey-Fuller test
adf_statistic, p_value





# Differencing the series
differenced_time_series = time_series.diff().dropna()

# Perform a Dickey-Fuller test again on the differenced series
result_diff = adfuller(differenced_time_series)
adf_statistic_diff = result_diff[0]
p_value_diff = result_diff[1]

# Display the results of the Dickey-Fuller test on the differenced data
adf_statistic_diff, p_value_diff




import itertools

# Define the p, d, and q parameters to take any value between 0 and 2
p = d = q = range(0, 3)
# Generate all different combinations of p, d, and q triplets
pdq = list(itertools.product(p, [1], q))  # d is fixed to 1 as we established that 1st differencing is needed

# Find the best ARIMA model based on AIC
best_aic = float("inf")
best_pdq = None
best_model = None

for param in pdq:
    try:
        temp_model = ARIMA(time_series, order=param)
        results = temp_model.fit()
        if results.aic < best_aic:
            best_aic = results.aic
            best_pdq = param
            best_model = results
    except:
        continue

# Show the best model's parameters and AIC
best_pdq, best_aic





# Correcting the forecast extraction and handling
forecast_result = best_model.get_forecast(steps=10)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Preparing the forecast dataframe for display, including confidence intervals
forecast_df = pd.DataFrame({
    'Forecasted Time In Ctrl. Vent': forecast,
    'Confidence Interval Lower': conf_int.iloc[:, 0],
    'Confidence Interval Upper': conf_int.iloc[:, 1]
})

forecast_df




import numpy as np

# Generate hypothetical new data (simulate realistic data based on the initial forecasts)
np.random.seed(42)  # for reproducibility
new_actual_data = forecast + np.random.normal(0, 200, 10)  # adding random noise with standard deviation of 200

# Append the new actual data to the original time series
updated_time_series = pd.concat([time_series, pd.Series(new_actual_data, index=np.arange(len(time_series), len(time_series) + 10))])

# Re-fit the ARIMA model to the updated dataset
updated_model = ARIMA(updated_time_series, order=best_pdq)
updated_results = updated_model.fit()

# Forecast the next 10 future values using the updated model
updated_forecast, updated_conf_int = updated_results.forecast(steps=10, alpha=0.05), updated_results.get_forecast(steps=10).conf_int()

# Prepare the forecast dataframe for display, including confidence intervals
updated_forecast_df = pd.DataFrame({
    'Updated Forecasted Time In Ctrl. Vent': updated_forecast,
    'Updated Confidence Interval Lower': updated_conf_int.iloc[:, 0],
    'Updated Confidence Interval Upper': updated_conf_int.iloc[:, 1]
})

updated_forecast_df




# Include the hypothetical actual data in the forecast DataFrame
updated_forecast_df['Hypothetical Actual Time In Ctrl. Vent'] = new_actual_data.values

# Display the updated forecast DataFrame with actual data included
updated_forecast_df





import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def retrain_and_forecast(file_path):
    # Load the new data from the file
    new_data = pd.read_excel(file_path)
    
    # Assume the relevant column names are 'Time In Ctrl. Vent' and 'Set Point Count'
    # Concatenate the new data with the existing data
    updated_time_series = pd.concat([existing_time_series, new_data['Time In Ctrl. Vent']])
    
    # Define the ARIMA model parameters from previous analysis or new analysis
    p, d, q = 1, 1, 0  # Example parameters, adjust based on prior tuning
    
    # Fit the ARIMA model to the updated dataset
    model = ARIMA(updated_time_series, order=(p, d, q))
    fitted_model = model.fit()
    
    # Forecast the next 10 values
    forecast, conf_int = fitted_model.get_forecast(steps=10), fitted_model.get_forecast(steps=10).conf_int()
    
    # Create a DataFrame to display the forecast and confidence intervals
    forecast_df = pd.DataFrame({
        'Forecasted Time In Ctrl. Vent': forecast.predicted_mean,
        'Confidence Interval Lower': conf_int.iloc[:, 0],
        'Confidence Interval Upper': conf_int.iloc[:, 1]
    })
    
    return forecast_df

# Example usage
file_path = 'path_to_your_new_data_file.xlsx'  # Update this path with your actual file path
forecast_results = retrain_and_forecast(file_path)
print(forecast_results)





import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def load_and_prepare_initial_data(file_path):
    # Load the initial dataset (this part is just for completeness and can be modified as needed)
    initial_data = pd.read_excel(file_path)
    return initial_data['Time In Ctrl. Vent']  # Assuming the time series column is named correctly

def retrain_and_forecast(new_data_path, initial_data):
    # Load the new data
    new_data = pd.read_excel(new_data_path)
    
    # Concatenate the new actual data with the initial dataset
    updated_time_series = pd.concat([initial_data, new_data['Time In Ctrl. Vent']])
    
    # Define and fit the ARIMA model (these parameters should be tuned as per your dataset)
    model = ARIMA(updated_time_series, order=(1, 1, 0))  # Using ARIMA(1,1,0) as an example
    fitted_model = model.fit()
    
    # Forecast the next 10 future values
    forecast_result = fitted_model.get_forecast(steps=10)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()
    
    # Create a DataFrame to display the forecast and confidence intervals
    forecast_df = pd.DataFrame({
        'Forecasted Time In Ctrl. Vent': forecast,
        'Confidence Interval Lower': conf_int.iloc[:, 0],
        'Confidence Interval Upper': conf_int.iloc[:, 1]
    })
    
    return forecast_df

# Example usage
initial_file_path = 'path_to_your_initial_data_file.xlsx'  # Update this path with your initial file path
initial_data = load_and_prepare_initial_data(initial_file_path)  # Load and prepare initial data

new_file_path = 'path_to_your_new_data_file.xlsx'  # Update this path with your new data file path
forecast_results = retrain_and_forecast(new_file_path, initial_data)
print(forecast_results)













