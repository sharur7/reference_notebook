from sklearn.cluster import KMeans

# Applying K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

# Adding the cluster labels to the dataframe
data['Volume Cluster KMeans'] = kmeans.labels_

# Summarizing the "Estimated Volume Adjusted" within each K-Means cluster
kmeans_cluster_summary = data.groupby('Volume Cluster KMeans')['Estimated Volume Adjusted'].describe()

# Visualizing the data points colored by their K-Means cluster
plt.figure(figsize=(12, 6))
sns.scatterplot(x=data.index, y='Estimated Volume Adjusted', hue='Volume Cluster KMeans', palette='viridis', data=data)
plt.title('Estimated Volume by K-Means Cluster')
plt.xlabel('Index')
plt.ylabel('Estimated Volume Adjusted')
plt.legend(title='Cluster')
plt.show()

kmeans_cluster_summary


# Identifying Cluster 0 and excluding the outlier (50) to calculate the mean
cluster_0_mean_without_outlier = data[(data['Volume Cluster KMeans'] == 0) & (data['Estimated Volume Adjusted'] != 50)]['Estimated Volume Adjusted'].mean()

# Replacing the outlier in Cluster 0 with the calculated mean
data.loc[(data['Volume Cluster KMeans'] == 0) & (data['Estimated Volume Adjusted'] == 50), 'Estimated Volume Adjusted'] = cluster_0_mean_without_outlier

# Recalculating the summary for Cluster 0 after adjustment
adjusted_cluster_0_summary = data[data['Volume Cluster KMeans'] == 0]['Estimated Volume Adjusted'].describe()

adjusted_cluster_0_summary


# Retry converting "Date and Time" without specifying a format, to handle any format inconsistencies automatically
data['Date and Time'] = pd.to_datetime(data['Date and Time'], errors='coerce')

# Recalculating the time difference between consecutive entries, now that "Date and Time" is properly formatted
data['Time Difference'] = data['Date and Time'].diff().dt.total_seconds().div(60)  # Convert seconds to minutes

# Identifying instances where the time difference exceeds 2 minutes
data['Potential Setup Change (Time)'] = data['Time Difference'] > 2

# Displaying instances where time difference exceeds 2 minutes
potential_setup_changes_time = data[data['Potential Setup Change (Time)'] == True][['Date and Time', 'Time Difference']]
potential_setup_changes_time


# Flagging rows where there's a cluster shift as potential setup changes
data['Cluster Shift'] = data['Volume Cluster KMeans'].diff().ne(0)

# Combining the criteria for setup changes into a single column
data['Setup Change'] = data['Potential Setup Change (Time)'] | data['Cluster Shift']

# Displaying the relevant columns to verify the setup change flags
setup_change_flags = data[['Date and Time', 'Volume Cluster KMeans', 'Time Difference', 'Potential Setup Change (Time)', 'Cluster Shift', 'Setup Change']]
setup_change_flags


# Filtering the data to only include rows where a setup change has been indicated
setup_change_data = data[data['Setup Change'] == True][['Date and Time', 'Set Point Value']]

setup_change_data



# Assuming 'data' is your DataFrame and 'Volume Cluster KMeans' identifies the clusters

# Step 2 & 3: Adjust the outlier value from 220 to 30 in Cluster 0
data.loc[(data['Volume Cluster KMeans'] == 0) & (data['Estimated Volume Adjusted'] == 220), 'Estimated Volume Adjusted'] = 30

# Step 4: Recalculate and display the summary statistics for Cluster 0
cluster_0_summary_after_adjustment = data[data['Volume Cluster KMeans'] == 0]['Estimated Volume Adjusted'].describe()
print(cluster_0_summary_after_adjustment)


# Assuming 'data' is your DataFrame and it includes a 'Volume Cluster KMeans' column for cluster labels

# Step 2: Adjust values over 30 to be exactly 30 in Cluster 0
data.loc[(data['Volume Cluster KMeans'] == 0) & (data['Estimated Volume Adjusted'] > 30), 'Estimated Volume Adjusted'] = 30

# Step 3: Verify the adjustment by recalculating the maximum value for Cluster 0
max_value_cluster_0 = data[data['Volume Cluster KMeans'] == 0]['Estimated Volume Adjusted'].max()
print(f"The adjusted maximum value for Cluster 0 is now: {max_value_cluster_0}")

