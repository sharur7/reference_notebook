from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Assuming df is your DataFrame containing the dataset

# Extract relevant features
X = df[['Set Point Value', 'Time In Ctrl. Vent']]
y = df['Time In Ctrl. Vent']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Prediction for the next day
# Assuming df_next_day contains data for the next day
X_next_day = df_next_day[['Set Point Value', 'Time In Ctrl. Vent']]
predicted_next_day = model.predict(X_next_day)

# Printing predicted control vent time for the next five set points
print('Predicted Control Vent Time for the Next Day:')
print(predicted_next_day[:5])


import matplotlib.pyplot as plt

# Assuming df_actual_next_day contains the actual data for the next day

# Visualization of actual vs predicted data
plt.figure(figsize=(10, 6))
plt.plot(df_actual_next_day.index, df_actual_next_day['Time In Ctrl. Vent'], label='Actual Control Vent Time')
plt.plot(df_next_day.index, predicted_next_day, label='Predicted Control Vent Time', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Control Vent Time')
plt.title('Actual vs Predicted Control Vent Time for Next Day')
plt.legend()
plt.show()






from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

class VentTimePredictor:
    def __init__(self):
        self.model = LinearRegression()

    def retrain(self, X_train, y_train):
        # Retrain the model with new data
        self.model.fit(X_train, y_train)

    def predict_and_visualize(self, X_next_day, df_actual_next_day):
        # Predict control vent time for the next day
        predicted_next_day = self.model.predict(X_next_day)
        
        # Visualization of actual vs predicted data
        plt.figure(figsize=(10, 6))
        plt.plot(df_actual_next_day.index, df_actual_next_day['Time In Ctrl. Vent'], label='Actual Control Vent Time')
        plt.plot(X_next_day.index, predicted_next_day, label='Predicted Control Vent Time', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Control Vent Time')
        plt.title('Actual vs Predicted Control Vent Time for Next Day')
        plt.legend()
        plt.show()

# Example usage
vent_predictor = VentTimePredictor()

# Assume df_train contains previous and current data for training
X_train = df_train[['Set Point Value', 'Time In Ctrl. Vent']]
y_train = df_train['Time In Ctrl. Vent']

# Initial training
vent_predictor.retrain(X_train, y_train)

# Assume df_next_day contains data for the next day
X_next_day = df_next_day[['Set Point Value', 'Time In Ctrl. Vent']]
df_actual_next_day = ...  # Actual data for the next day

# Prediction and visualization
vent_predictor.predict_and_visualize(X_next_day, df_actual_next_day)

# Assume df_new_data contains new data for the next day (including previous and current data)
# Retrain with the updated dataset
vent_predictor.retrain(df_new_data[['Set Point Value', 'Time In Ctrl. Vent']], df_new_data['Time In Ctrl. Vent'])

# Prediction and visualization with retrained model
vent_predictor.predict_and_visualize(X_next_day, df_actual_next_day)





