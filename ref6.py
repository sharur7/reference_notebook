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


# Simplified approach to identify and compare similar setpoint sequences
# Step 1: Extend the cycle identification to include the sequence of set points

def get_setpoint_sequence(df):
    """Extract setpoint sequences for each calibration cycle."""
    df['Set Point Sequence'] = df.groupby('Calibration Cycle')['Set Point Value'].transform(lambda x: ','.join(x.astype(str)))
    return df

data_with_sequences = get_setpoint_sequence(data_with_cycles)

# Step 2: Identify unique setpoint sequences and tag cycles for comparison
unique_sequences = data_with_sequences['Set Point Sequence'].drop_duplicates().reset_index(drop=True)
sequence_to_tag = {seq: idx for idx, seq in unique_sequences.iteritems()}
data_with_sequences['Sequence Tag'] = data_with_sequences['Set Point Sequence'].map(sequence_to_tag)

# Now, analyze time parameters for cycles with matching sequences
# Aggregating mean time parameters by sequence tag for comparison
sequence_grouped_data = data_with_sequences.groupby('Sequence Tag').agg({
    'Time In Ctrl. Vent': 'mean',
    'Time In Cent. Vent': 'mean',
    'Time Centering': 'mean'
}).reset_index()

sequence_grouped_data




# Grouping the data by 'Set Point Value' and calculating mean for the relevant time-related parameters
grouped_data = data.groupby('Set Point Value').agg({
    'Time In Ctrl. Vent': 'mean',
    'Time Centering': 'mean'
}).reset_index()

# Renaming columns for clarity
grouped_data.rename(columns={
    'Time In Ctrl. Vent': 'Average Time In Ctrl. Vent',
    'Time Centering': 'Average Time In Centering'
}, inplace=True)

grouped_data.head()


# Calculating standard deviation for the relevant time-related parameters by 'Set Point Value'
std_data = data.groupby('Set Point Value').agg({
    'Time In Ctrl. Vent': 'std',
    'Time Centering': 'std'
}).reset_index()

# Renaming columns for clarity
std_data.rename(columns={
    'Time In Ctrl. Vent': 'Std. Dev. Time In Ctrl. Vent',
    'Time Centering': 'Std. Dev. Time In Centering'
}, inplace=True)

std_data.head()




# Replacing NaN values with 0 to indicate no variability
std_data_filled = std_data.fillna(0)

std_data_filled.head()


# Defining the threshold value
threshold_value = 5

# Classifying set points as 'Problematic' or 'Not Problematic' based on the specified threshold
std_data_filled['Problematic'] = ((std_data_filled['Std. Dev. Time In Ctrl. Vent'] > threshold_value) | 
                                  (std_data_filled['Std. Dev. Time In Centering'] > threshold_value)).map({True: 'Problematic', False: 'Not Problematic'})

std_data_filled.head()


# Filtering the dataset to include only the necessary columns for this analysis
comparison_data = data[['Set Point Value', 'Time In Ctrl. Vent', 'Time Centering']].copy()

# Defining a function to compare records within each group and classify based on specified criteria
def classify_set_points(group):
    # Initialize a DataFrame to hold comparison results
    comparison_results = pd.DataFrame(columns=['Time In Ctrl. Vent Difference', 'Time Centering Difference', 'Problematic'])
    
    # Iterate over all combinations of records within the group for comparison
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            # Calculate the absolute differences for the two time parameters
            ctrl_vent_diff = abs(group.iloc[i]['Time In Ctrl. Vent'] - group.iloc[j]['Time In Ctrl. Vent'])
            centering_diff = abs(group.iloc[i]['Time Centering'] - group.iloc[j]['Time Centering'])
            
            # Classify as problematic if any difference exceeds the threshold of 4
            problematic = 'Problematic' if ctrl_vent_diff > 4 or centering_diff > 4 else 'Not Problematic'
            
            # Append the comparison result to the DataFrame
            comparison_results = comparison_results.append({
                'Time In Ctrl. Vent Difference': ctrl_vent_diff,
                'Time Centering Difference': centering_diff,
                'Problematic': problematic
            }, ignore_index=True)
            
    return comparison_results

# Apply the comparison and classification function to each group of identical set point values
problematic_classification = comparison_data.groupby('Set Point Value').apply(classify_set_points).reset_index(drop=True)

# Since the function could potentially generate a large number of comparisons, let's view the summary of problematic classifications
problematic_summary = problematic_classification['Problematic'].value_counts()

problematic_summary


# Calculating the max and min for both time parameters within each set point value group
max_min_diff = data.groupby('Set Point Value').agg({
    'Time In Ctrl. Vent': ['max', 'min'],
    'Time Centering': ['max', 'min']
}).reset_index()

# Calculating the difference between max and min for each parameter
max_min_diff[('Time In Ctrl. Vent', 'Difference')] = max_min_diff[('Time In Ctrl. Vent', 'max')] - max_min_diff[('Time In Ctrl. Vent', 'min')]
max_min_diff[('Time Centering', 'Difference')] = max_min_diff[('Time Centering', 'max')] - max_min_diff[('Time Centering', 'min')]

# Identifying problematic set points where the difference exceeds 4 units
max_min_diff['Problematic'] = ((max_min_diff[('Time In Ctrl. Vent', 'Difference')] > 4) | 
                               (max_min_diff[('Time Centering', 'Difference')] > 4)).map({True: 'Problematic', False: 'Not Problematic'})

# Simplifying the DataFrame for display
simplified_problematic_classification = max_min_diff[['Set Point Value', ('Time In Ctrl. Vent', 'Difference'), ('Time Centering', 'Difference'), 'Problematic']]

simplified_problematic_classification.head()




def detect_all_cycles(Setpoints):
    cycles = []
    for start_index in range(len(Setpoints)):
        for end_index in range(start_index + 1, len(Setpoints)):
            if Setpoints[start_index] == Setpoints[end_index]:
                # Check if the detected cycle overlaps with any previously detected cycle
                overlap = any(start_index < cycle[1] and end_index > cycle[0] for cycle in cycles)
                if not overlap:
                    cycles.append((start_index, end_index))
                    break  # Move to the next start_index after finding a non-overlapping cycle

    return cycles

# Apply the improved function to the 'Set Point Value' column of the updated dataframe
all_detected_cycles = detect_all_cycles(updated_df['Set Point Value'].tolist())

# Display the first few detected cycles from the updated dataset
all_detected_cycles[:5]

