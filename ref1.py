import pandas as pd

# Load the data from the uploaded file
file_path = '/mnt/data/updated_combined_data.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()


import numpy as np

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


