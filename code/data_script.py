import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# Metadata and variables
metadata = statlog_german_credit_data.metadata
variables = statlog_german_credit_data.variables

# Combine `y` with `X`
df = pd.concat([y, X], axis=1)
# Rename the target column using metadata
y_column_name = metadata['target_col'][0]  # Extract the first (and only) target column name
df.rename(columns={y.name: y_column_name}, inplace=True)

# Map value labels to the target variable, if available
if y_column_name in variables and 'values' in variables[y_column_name]:
    value_labels = variables[y_column_name]['values']
    df[y_column_name] = df[y_column_name].map(value_labels)

# Rename the columns with the following vector


# Display resulting dataset
print(df.head())