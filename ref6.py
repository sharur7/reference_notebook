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

