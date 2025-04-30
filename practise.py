import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(0)

# Example: Create a DataFrame with random values
num_rows = 10
num_cols = 5

# Generate random data
data = np.random.randn(num_rows, num_cols)  # Random normal distribution data

# Create column names
columns = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5']

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Print the DataFrame
print(df)
