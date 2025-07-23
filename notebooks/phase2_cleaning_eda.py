import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("images", exist_ok=True)

# Load data
df = pd.read_csv("data/telco_churn.csv")

# step 1 : Convert TotalCharges to numeric
print("Before: ",df['TotalCharges'].dtype)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("After: ",df['TotalCharges'].dtype)

# Check rows with NaN
print("\nRows with missing TotalCharges:")
print(df[df['TotalCharges'].isna()])

# Drop rows wjere Total Charges is NaN
df = df.dropna(subset=["TotalCharges"])
df.reset_index(drop=True, inplace=True)

# ----- STEP 2: Target column distribution -----
print("\nChurn distribution:")
print(df['Churn'].value_counts())

# Plot Class distribution
sns.countplot(x="Churn",hue="gender", data=df)
plt.title("Churn Count")
plt.savefig("images/churn_count.png")
print("Generating churn count plot...")

plt.show()

# Step 3 Churn vs Contract Type

sns.countplot(x = 'Contract', hue = 'Churn', data = df)
plt.title("Churn Count by Contract Type")
plt.savefig("images/churn_count_by_contract.png")
print("Generating churn count plot...")

plt.show()

# Step 4 : Monthly charges vs churn

sns.boxplot(x='Churn', y = 'MonthlyCharges', data = df)
plt.title("monthly Charges vs churn")
plt.savefig("images/monthly_charges_vs_churn.png")
print("Generating churn count plot...")

plt.show()

# optinal: save cleaned version

df.to_csv("data/cleaned_telco.csv", index=False)