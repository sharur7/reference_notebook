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


