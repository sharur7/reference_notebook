# Grouping the data by 'Set Point Value' and calculating the average for the relevant time parameters
grouped_data = data.groupby('Set Point Value').agg({
    'Time In Ctrl. Vent': 'mean',
    'Time In Cent. Vent': 'mean',
    'Time Centering': 'mean'
}).reset_index()

# Since the data might contain many unique set point values, we'll display a summary and then some of the grouped data
summary = grouped_data.describe()
sample_grouped_data = grouped_data.head(10)  # Displaying the first 10 rows as a sample

summary, sample_grouped_data
