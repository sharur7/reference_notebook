from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare the data for modeling
X = data_filtered.drop('Time In Ctrl. Vent', axis=1)
y = data_filtered['Time In Ctrl. Vent']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the root mean squared error for the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Return the model's performance
rmse, y_pred, y_test.values














import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def update_and_forecast(new_data_row, data, order=(1,1,1)):
    # Append new data
    data = pd.concat([data, pd.DataFrame([new_data_row], columns=data.columns)], ignore_index=True)
    
    # Extract target variable
    y = data['Time In Ctrl. Vent']
    
    # Fit ARIMA model
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    
    # Forecast the next day
    next_day_forecast = model_fit.forecast(steps=1)
    
    return next_day_forecast, model_fit, data

# Initial data load (assuming data is already loaded in 'data_filtered')
data = data_filtered.copy()

# New data for a day simulated as an example
new_data_example = {
    'Set Point Value': -450.10,
    'Distance Travelled': 6910,
    'Battery Capacity': 5700,
    'Board Temperature': 29.5,
    'Time Fast Vent': 6000,
    'Time In Ctrl. Vent': 1600  # This would be known at the end of the day
}

# Call the function with new data
forecast, updated_model, updated_data = update_and_forecast(new_data_example, data)

print("Next Day Forecast:", forecast)







# Example setup for continuous forecasting and model updating
def continuous_forecast_update(data, initial_size, order=(1,1,1)):
    forecasted = []
    actual = []
    indices = []
    
    # Loop over the data, simulating daily updates
    for i in range(initial_size, len(data)):
        # Current data up to day i
        current_data = data.iloc[:i]
        y = current_data['Time In Ctrl. Vent']
        
        # Fit ARIMA model
        model = ARIMA(y, order=order)
        model_fit = model.fit()
        
        # Forecast the next day
        next_day_forecast = model_fit.forecast(steps=1)
        
        # Store forecast and actual value
        forecasted.append(next_day_forecast.iloc[0])
        actual.append(data.iloc[i]['Time In Ctrl. Vent'])
        indices.append(data.index[i])
        
    return indices, forecasted, actual

# Using a subset of data for demonstration
initial_size = int(len(data_filtered) * 0.8)  # 80% for initial training

# Simulate the process
indices, forecasted, actual = continuous_forecast_update(data_filtered, initial_size)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(indices, actual, 'r-', label='Actual Time In Ctrl. Vent')
plt.plot(indices, forecasted, 'b--', label='Forecasted Time In Ctrl. Vent')
plt.title('Continuous Forecasting and Model Updating')
plt.xlabel('Date and Time')
plt.ylabel('Time In Ctrl. Vent')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

