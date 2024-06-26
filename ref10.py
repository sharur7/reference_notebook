import matplotlib.pyplot as plt

# Filter data up to the 3rd day
data_until_3rd = data[data['Date and Time'].dt.day <= 3]

# Plotting "Time In Ctrl. Vent"
plt.figure(figsize=(12, 6))
plt.plot(data_until_3rd['Date and Time'], data_until_3rd['Time In Ctrl. Vent'], marker='o', linestyle='-')
plt.title('Time In Control Vent over Time')
plt.xlabel('Date and Time')
plt.ylabel('Time In Ctrl. Vent')
plt.grid(True)
plt.show()


from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# Ensure the data is sorted by date
data_until_3rd = data_until_3rd.sort_values(by='Date and Time')

# Preparing the data for the ARIMA model
time_in_ctrl_vent_series = data_until_3rd['Time In Ctrl. Vent'].astype(np.float64)

# Define the ARIMA model, let's start with a simple ARIMA(1,1,1)
model = ARIMA(time_in_ctrl_vent_series, order=(1,1,1))

# Fit the model
fitted_model = model.fit()

# Forecast the next step, which would correspond to the 4th
forecast = fitted_model.forecast(steps=1)

forecast







import numpy as np

# Assume we have hourly data points and we generate hypothetical data for 24 hours on the 4th day
hours = np.arange(24)
forecasted_values = np.full(24, 150)  # Assume the model predicted a constant value of 150
actual_values = forecasted_values + np.random.normal(0, 10, 24)  # Actual values vary around the forecasted with some noise

# Creating a DataFrame for easier plotting
comparison_df = pd.DataFrame({
    'Hour': hours,
    'Forecasted': forecasted_values,
    'Actual': actual_values
})

# Plotting both forecasted and actual values
plt.figure(figsize=(12, 6))
plt.plot(comparison_df['Hour'], comparison_df['Forecasted'], label='Forecasted', marker='o')
plt.plot(comparison_df['Hour'], comparison_df['Actual'], label='Actual', marker='x')
plt.title('Comparison of Forecasted vs Actual Values for the 4th Day')
plt.xlabel('Hour of Day')
plt.ylabel('Time In Ctrl. Vent')
plt.legend()
plt.grid(True)
plt.show()
