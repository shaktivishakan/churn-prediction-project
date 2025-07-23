# Phase 3: Feature Engineering

import pandas as pd

# Load Cleaned data from phase2
df = pd.read_csv("data/cleaned_telco.csv")

# Step: 1 Drop unnecessary Columns
df = df.drop(["customerID"], axis=1)

# Step : 2  Encode Target 

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Step: 3   Indentify Categorical Columns

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categrorical columns:", cat_cols)

# Step: 4   One-Hot enocde categorical colums
df = pd.get_dummies(df, columns=cat_cols, drop_first = True)

# Step: 5   Save Final dataset for modelling
df.to_csv("data/processed_telco.csv", index=False)

print("\nFinal dataset shape:\n", df.shape)
