import pandas as pd

# Load data
df = pd.read_csv("data/telco_churn.csv")

# shape first few rows
print("Dataset shape:\n", df.shape)
print("\nFirst 5 rows:\n", df.head())

# check for null values
print("\nNull values:\n", df.isnull().sum())

#check data types
print("\nData info:\n", df.info())

#check for duplicates
print("\nDuplicate values:\n", df.duplicated().sum())

# describe numerical Columns
print("\nSummary Statistics:\n", df.describe())