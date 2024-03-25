import seaborn as sns

# Visualization: "Time To Stable" vs. "Set Point Value" with a trend line
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data['Set Point Value'], y=data['Time To Stable'], alpha=0.6)
sns.regplot(x=data['Set Point Value'], y=data['Time To Stable'], scatter=False, lowess=True, color='red', label='Trend Line')
plt.title('Time To Stable vs. Set Point Value with Trend')
plt.xlabel('Set Point Value')
plt.ylabel('Time To Stable')
plt.legend()
plt.grid(True)
plt.show()


# Step 1: Detect setpoint changes
# We use shift to compare each row to the next, identifying where the set point changes
data['Set Point Changed'] = data['Set Point Value'] != data['Set Point Value'].shift(1)

# Step 2: Calculate the time difference between each set point change
# First, ensure 'Date and Time' is in datetime format for time difference calculation
data['Date and Time'] = pd.to_datetime(data['Date and Time'])

# Calculate the time difference in minutes
data['Time Diff'] = data['Date and Time'].diff().dt.total_seconds().div(60).fillna(0)

# Step 3: Flag setup changes
# A setup change is identified where the set point changes and the time difference exceeds 2 minutes
data['Setup Change'] = (data['Set Point Changed']) & (data['Time Diff'] > 2)

# Extracting rows where a setup change is detected
setup_changes = data[data['Setup Change']]

setup_changes[['Date and Time', 'Set Point Value', 'Time Diff', 'Setup Change']].head()

# Adjusting the criteria for identifying setup changes
data['Setup Change'] = (data['Set Point Changed']) | (data['Time Diff'] > 2)

# Extracting rows where a setup change is detected under the updated criteria
setup_changes_adjusted = data[data['Setup Change']]

setup_changes_adjusted[['Date and Time', 'Set Point Value', 'Time Diff', 'Setup Change']].head()


# Step 1: Detect volume changes
# We use shift to compare each row to the next, identifying where the volume changes
data['Volume Changed'] = data['Estimated Volume'] != data['Estimated Volume'].shift(1)

# Step 2 and 3: Use the existing 'Time Diff' column to flag setup changes based on volume changes
data['Setup Change Volume'] = (data['Volume Changed']) & (data['Time Diff'] > 2)

# Extracting rows where a setup change is detected based on volume change
setup_changes_volume = data[data['Setup Change Volume']]

setup_changes_volume[['Date and Time', 'Estimated Volume', 'Time Diff', 'Setup Change Volume']].head()


# Adjusting the criteria for identifying setup changes based on volume changes
data['Setup Change Volume'] = (data['Volume Changed']) | (data['Time Diff'] > 2)

# Extracting rows where a setup change is detected under the updated criteria
setup_changes_volume_adjusted = data[data['Setup Change Volume']]

setup_changes_volume_adjusted[['Date and Time', 'Estimated Volume', 'Time Diff', 'Setup Change Volume']].head()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Feature selection
features = ['Set Point Value', 'Estimated Volume', 'Time Diff']
X = data[features]

# The label is whether a setup change occurred based on volume or time difference
y = data['Setup Change Volume'].astype(int)

# Preprocess the Data: Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the Data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Selection and Training: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)

# Evaluation
evaluation_report = classification_report(y_test, y_pred)

evaluation_report

def detect_calibration_cycles(set_point_values):
    """
    Detects calibration cycles based on set point values.
    A cycle is defined as a sequence starting and ending at the same set point value, not including the starting point itself.
    """
    cycles = []  # List to hold the start and end indices of each cycle
    start_index = None  # Starting index of a potential cycle
    
    for i, value in enumerate(set_point_values):
        if start_index is None:
            # Potential start of a new cycle
            start_index = i
        elif value == set_point_values[start_index]:
            # End of a cycle, append start and end indices to the cycles list
            if i > start_index:  # Ensures the cycle includes more than just the starting point
                cycles.append((start_index, i))
                start_index = None  # Reset for the next cycle
    
    return cycles

# Detect calibration cycles in the Set Point Value column
calibration_cycles_indices = detect_calibration_cycles(data['Set Point Value'])

# Summary of detected cycles
cycles_summary = {
    'Number of Cycles': len(calibration_cycles_indices),
    'Cycle Indices': calibration_cycles_indices
}

cycles_summary

def detect_calibration_cycles(set_point_values):
    """
    Detect calibration cycles based on the set point values. 
    A cycle starts and ends at the same point and has at least one different value in between.

    Parameters:
    - set_point_values: pandas Series of set point values.

    Returns:
    - cycles: List of tuples, each representing a cycle (start_index, end_index).
    """
    cycles = []
    cycle_start = None

    for i in range(1, len(set_point_values)):
        # Start of a new cycle
        if cycle_start is None and set_point_values[i] != set_point_values[i-1]:
            cycle_start = i - 1

        # End of a cycle
        elif cycle_start is not None and set_point_values[i] == set_point_values[cycle_start]:
            cycles.append((cycle_start, i))
            cycle_start = None  # Reset for the next cycle

    return cycles

# Detect calibration cycles
calibration_cycles = detect_calibration_cycles(data['Set Point Value'])

# Display the first few cycles to verify the method is working as expected
calibration_cycles[:5]
