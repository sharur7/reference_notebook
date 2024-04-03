df['Normalized SPV Difference'] = (df['Set Point Value Difference'] / df['Set Point Value'].shift(1)) * 100

# Classifying 'Set Point Value Difference' according to the specified thresholds
def classify_difference(difference):
    if difference > 1500:
        return 'Large'
    elif 200 < difference <= 1500:
        return 'Medium'
    else:
        return 'Small'

# Applying the classification function to the 'Set Point Value Difference' column
df['Classification'] = df['Set Point Value Difference'].apply(classify_difference)

# Display the dataframe to confirm the classification
df[['Date and Time', 'Set Point Value Difference', 'Classification']].head()
