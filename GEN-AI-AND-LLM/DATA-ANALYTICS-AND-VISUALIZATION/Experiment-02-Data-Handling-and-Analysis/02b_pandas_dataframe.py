# EX.NO: 2B - Working with Pandas DataFrames
# AIM: Perform data manipulation operations using Pandas

import pandas as pd
import numpy as np

# Create sample DataFrame
data = {
    'Name':  ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age':   [25, 30, 35, 28, 22],
    'Score': [88.5, 92.0, 78.3, 95.1, 85.6],
    'City':  ['Chennai', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai']
}
df = pd.DataFrame(data)

print("--- DataFrame ---")
print(df)
print("\n--- Info ---")
df.info()
print("\n--- Summary Statistics ---")
print(df.describe())

# Handle missing values
df_nan = df.copy()
df_nan.loc[1, 'Score'] = None
print("\n--- DataFrame with NaN ---")
print(df_nan)
df_nan.fillna(df_nan['Score'].mean(), inplace=True)
print("\n--- After Filling NaN ---")
print(df_nan)

# Add computed column
df['Grade'] = df['Score'].apply(
    lambda x: 'A' if x >= 90 else ('B' if x >= 80 else 'C')
)
print("\n--- With Grade Column ---")
print(df)

# Filtering
print("\n--- Students with Score > 85 ---")
print(df[df['Score'] > 85])

# Groupby
print("\n--- Mean Score by City ---")
print(df.groupby('City')['Score'].mean())

# Sort
print("\n--- Sorted by Score (descending) ---")
print(df.sort_values('Score', ascending=False))

# Save to CSV
df.to_csv('processed_data.csv', index=False)
print("\nSaved to processed_data.csv")
