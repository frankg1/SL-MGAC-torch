import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data_file = 'rl_model_train_data_for_torch_1011_cleaned.txt'
train_file = 'rl_model_train_data_for_torch_1011_cleaned_train.txt'
test_file = 'rl_model_train_data_for_torch_1011_cleaned_test.txt'

# Read the data
df = pd.read_csv(
    data_file,
    sep='\x01',
    header=None,
    na_values='\\N',
)

# Extract the first column (device_id)
device_ids = df[1].unique()  # Get unique device_ids
print('device_id: ',device_ids[0])

# Perform a train-test split on the device_ids
train_ids, test_ids = train_test_split(device_ids, test_size=0.2, random_state=42)

# Split the original DataFrame based on the device_ids
train_data = df[df[1].isin(train_ids)]
test_data = df[df[1].isin(test_ids)]

# Save the train data
train_data.to_csv(
    train_file,
    sep='\x01',
    index=False,    # Do not save row indices
    header=False,   # Do not save column headers
    na_rep='\\N'    # Represent missing values as '\\N'
)

# Save the test data
test_data.to_csv(
    test_file,
    sep='\x01',
    index=False,    # Do not save row indices
    header=False,   # Do not save column headers
    na_rep='\\N'    # Represent missing values as '\\N'
)

print('data set size:', train_data.shape[0], test_data.shape[0])