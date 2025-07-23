import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load processed data and model
df = pd.read_csv("data/processed_telco.csv")
model = joblib.load("models/xgb_model.joblib")

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# -------------------------------
# SHAP Analysis
# -------------------------------

# 1. Create TreeExplainer
explainer = shap.Explainer(model)

# 2. Calculate SHAP values
shap_values = explainer(X)

# 3. Summary Plot (Feature Importance)
shap.summary_plot(shap_values, X)

# 4. Optional: Individual prediction explanation
# Pick one customer
i = 10
shap.plots.waterfall(shap_values[i], max_display=10)

# Save if needed
# plt.savefig("images/shap_summary.png")
