# EX.NO: 2C - Reading Data from Text Files and Web
# AIM: Read data from CSV files and web URLs using Pandas

import pandas as pd

# --- Read from CSV (Text File) ---
print("=== Reading from CSV ===")
try:
    df = pd.read_csv('data.csv')
    print(df.head())
except FileNotFoundError:
    print("Creating sample CSV...")
    sample = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [90, 85, 78]
    })
    sample.to_csv('data.csv', index=False)
    df = pd.read_csv('data.csv')
    print(df)

# Fill missing values
df.ffill(inplace=True)
df.to_csv('processed_text.csv', index=False)
print("\nProcessed data saved.")

# --- Read from Web ---
print("\n=== Reading from Web ===")
try:
    url = 'https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv'
    web_df = pd.read_csv(url)
    print("Columns:", list(web_df.columns))
    print(web_df.head())
except Exception as e:
    print(f"Web read failed (network): {e}")
    print("Creating sample web-like data locally...")
    web_sample = pd.DataFrame({
        'Country': ['India', 'USA', 'UK'],
        'Population': [1380, 331, 67]
    })
    print(web_sample)
