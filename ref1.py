import pandas as pd

# Load the data from the uploaded file
file_path = '/mnt/data/updated_combined_data.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


import numpy as np



# Feature Engineering: Creating new features for venting rate

# We might use 'Time Fast Vent', 'Time Coarse Control', and 'Time Fine Control' as indicators of venting time
# We'll create a feature for the rate of venting - this could be time of venting over distance travelled
data['Rate Fast Vent'] = data['Time Fast Vent'] / data['Distance Travelled']
data['Rate Coarse Control'] = data['Time Coarse Control'] / data['Distance Travelled']
data['Rate Fine Control'] = data['Time Fine Control'] / data['Distance Travelled']

# Replace infinite values with NaN
data.replace([float('inf'), -float('inf')], np.nan, inplace=True)

# Descriptive statistics for these new features
rate_features_descriptive_stats = data[['Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']].describe()

rate_features_descriptive_stats


# Replacing infinite values with NaN
data['Rate Fast Vent'] = data['Rate Fast Vent'].replace([np.inf, -np.inf], np.nan)
data['Rate Coarse Control'] = data['Rate Coarse Control'].replace([np.inf, -np.inf], np.nan)
data['Rate Fine Control'] = data['Rate Fine Control'].replace([np.inf, -np.inf], np.nan)

# Descriptive statistics for these new features
rate_features_descriptive_stats = data[['Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']].describe()
rate_features_descriptive_stats


# Calculate mean and standard deviation for each venting rate feature
mean_std = rate_features_descriptive_stats.loc[['mean', 'std']]

# Defining thresholds for anomalies (using 3 standard deviations here)
thresholds = mean_std.loc['mean'] + 3 * mean_std.loc['std']

# Identifying anomalies
anomalies = data[(data['Rate Fast Vent'] > thresholds['Rate Fast Vent']) | 
                 (data['Rate Coarse Control'] > thresholds['Rate Coarse Control']) | 
                 (data['Rate Fine Control'] > thresholds['Rate Fine Control'])]

# Output the thresholds and the anomalies identified
thresholds, anomalies[['Date and Time', 'Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']]



from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Preprocessing: Normalizing the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']].fillna(0))

# Apply Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomalies_iso_forest = iso_forest.fit_predict(scaled_data)

# Adding the results back to the original dataframe
data['Anomaly_IsoForest'] = anomalies_iso_forest

# Identifying the rows marked as anomalies (-1 indicates an anomaly)
anomaly_data_iso_forest = data[data['Anomaly_IsoForest'] == -1]

# Displaying the anomalies
anomaly_data_iso_forest[['Date and Time', 'Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']]



from sklearn.model_selection import train_test_split

# Defining the target variable based on the identified anomalies
# Anomalies (detected by Isolation Forest) are marked as '1', normal points as '0'
data['Target_Failure'] = data['Anomaly_IsoForest'].apply(lambda x: 1 if x == -1 else 0)

# Selecting features for the model
# For this example, we're using the venting rate features and other relevant operational parameters
# Note: More features can be added based on domain knowledge and data availability
features = ['Set Point Value', 'Distance Travelled', 'Battery Capacity', 'Board Temperature',
            'Rate Fast Vent', 'Rate Coarse Control', 'Rate Fine Control']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data[features], data['Target_Failure'], test_size=0.3, random_state=42)

# Displaying the shapes of the splits to verify
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Training the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Making predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

accuracy, confusion_mat, classification_rep




# Assuming specific setpoint values are not provided, we will look for notable patterns
# Typically, in hysteresis testing, setpoints might alternate between high and low values

# A simple approach to identify potential calibration cycles:
# 1. Find large changes in the set point value that might indicate a switch between high and low values
# 2. Look at the frequency and regularity of these changes

# Calculating changes in setpoint values
data['Set Point Change'] = data['Set Point Value'].diff().abs()

# Assuming a significant change threshold (this could be adjusted based on domain knowledge)
significant_change_threshold = data['Set Point Change'].mean() + data['Set Point Change'].std()

# Identifying potential calibration cycles
potential_calibration_cycles = data[data['Set Point Change'] > significant_change_threshold]

# Displaying potential calibration cycles
potential_calibration_cycles[['Date and Time', 'Set Point Value', 'Set Point Change']]

\

# Reattempting the analysis with a different approach

# Calculating the absolute change in setpoint values between consecutive measurements
data['Set Point Change'] = data['Set Point Value'].diff().abs()

# Assuming a significant change threshold
# This threshold can be adjusted based on domain knowledge or specific criteria for calibration cycles
significant_change_threshold = data['Set Point Change'].mean() + data['Set Point Change'].std()

# Identifying rows where the setpoint change is significant, indicating potential calibration cycles
potential_calibration_cycles = data[data['Set Point Change'] > significant_change_threshold]

# Displaying potential calibration cycles with their corresponding setpoint values and changes
potential_calibration_cycles[['Date and Time', 'Set Point Value', 'Set Point Change']].head()  # Showing the top rows


# Directly inspecting the setpoint values to spot potential calibration cycles
# Displaying a portion of the setpoint values to observe any notable patterns
setpoint_values = data[['Date and Time', 'Set Point Value']]
setpoint_values.head(20)  # Displaying the first 20 rows for inspection

import matplotlib.pyplot as plt

# Plotting the Set Point Values to visually inspect for calibration cycles
plt.figure(figsize=(15, 6))
plt.plot(data['Date and Time'], data['Set Point Value'], marker='o')
plt.title('Set Point Values Over Time')
plt.xlabel('Date and Time')
plt.ylabel('Set Point Value')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


from scipy.signal import find_peaks

# Identifying peaks (maxima) and troughs (minima) in the 'Set Point Value'
peaks, _ = find_peaks(data['Set Point Value'])
troughs, _ = find_peaks(-data['Set Point Value'])

# Plotting the identified peaks and troughs
plt.figure(figsize=(15, 6))
plt.plot(data['Date and Time'], data['Set Point Value'], label='Set Point Value')
plt.scatter(data['Date and Time'][peaks], data['Set Point Value'][peaks], color='red', label='Peaks')
plt.scatter(data['Date and Time'][troughs], data['Set Point Value'][troughs], color='green', label='Troughs')
plt.title('Set Point Value Over Time with Peaks and Troughs')
plt.xlabel('Date and Time')
plt.ylabel('Set Point Value')
plt.grid(True)
plt.legend()
plt.show()


def detect_calibration_cycles(data, column='Set Point Value', threshold=0.1):
    cycles = []
    increasing = False
    start_index = None

    for i in range(1, len(data)):
        if start_index is None:
            # Start of a potential cycle
            if data[column][i] > data[column][i-1]:
                increasing = True
                start_index = i-1
        else:
            if increasing and data[column][i] < data[column][i-1]:
                # Switch from increasing to decreasing
                increasing = False
            elif not increasing and (data[column][i] > data[column][i-1] or
                                     abs(data[column][start_index] - data[column][i]) <= threshold):
                # End of the cycle
                end_index = i
                cycles.append((data['Date and Time'][start_index], data['Date and Time'][end_index]))
                start_index = None
                increasing = False

    return cycles

# Detecting calibration cycles
calibration_cycles = detect_calibration_cycles(data)

# Displaying the first few detected cycles
calibration_cycles[:5]



import pandas as pd

def find_calibration_cycles(set_point_values, threshold=100):
    """
    Find calibration cycles in the set point values.

    Args:
    - set_point_values (list): The list of set point values.
    - threshold (float): The value difference considered significant for starting a cycle.

    Returns:
    - list of tuples: Each tuple contains the start and end indices of a calibration cycle.
    """
    cycles = []
    i = 0
    while i < len(set_point_values) - 1:
        # Find start of potential cycle
        if abs(set_point_values[i+1] - set_point_values[i]) > threshold:
            cycle_start = i
            peak_found = False
            # Find peak of cycle
            for j in range(i + 1, len(set_point_values) - 1):
                if (set_point_values[j] > set_point_values[j - 1] and set_point_values[j] > set_point_values[j + 1]) or \
                   (set_point_values[j] < set_point_values[j - 1] and set_point_values[j] < set_point_values[j + 1]):
                    peak_found = True
                elif peak_found and abs(set_point_values[j] - set_point_values[cycle_start]) < threshold:
                    # End of cycle found
                    cycles.append((cycle_start, j))
                    i = j - 1  # Update outer loop index to continue after current cycle
                    break
        i += 1
    return cycles

# Example usage:
# Load your dataset
# df = pd.read_excel("path_to_your_file.xlsx")
# set_point_values = df['Set Point Value'].tolist()

# Find calibration cycles
# calibration_cycles = find_calibration_cycles(set_point_values)
# print(calibration_cycles)
