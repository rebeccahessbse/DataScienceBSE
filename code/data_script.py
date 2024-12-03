import pandas as pd
from ucimlrepo import fetch_ucirepo
import re

# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# Metadata and variables
metadata = statlog_german_credit_data.metadata
variables = statlog_german_credit_data.variables
variables_info = metadata['additional_info']['variable_info']

# Combine `y` with `X`
df = pd.concat([y, X], axis=1)
df.rename(columns={df.columns[0]: 'class'}, inplace=True)

def parse_metadata_info(variable_info):
    mappings = {}
    attributes = variable_info.split("Attribute")
    for attribute in attributes:
        if not attribute.strip():
            continue
        lines = attribute.splitlines()
        # The first line contains the attribute number and type
        if len(lines) > 1 and "(qualitative)" in lines[0]:
            attr_num = lines[0].strip().split(":")[0]
            column_name = f"Attribute{attr_num}"
            # Extract mappings from the remaining lines
            values = {}
            for line in lines[1:]:
                line = line.strip()
                if ":" in line:
                    code, description = line.split(":", 1)
                    values[code.strip()] = description.strip()
            mappings[column_name] = values
    return mappings


# Parse metadata for mappings
attribute_mappings = parse_metadata_info(variables_info)

# Apply mappings to the DataFrame
for col, mapping in attribute_mappings.items():
    if col in df.columns:  # Ensure the column exists in the DataFrame
        df[col] = df[col].map(mapping)

# Display the updated DataFrame
print(df.head())
