# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load("models/xgb_model.joblib")
df = pd.read_csv("data/processed_telco.csv")
X = df.drop("Churn", axis=1)

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction")

# Option 1: Pick a random customer
if st.button("🎲 Pick a random customer"):
    sample = X.sample(1)
    st.write("### Sample Input Data:")
    st.dataframe(sample)

    # Prediction
    prob = model.predict_proba(sample)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"

    st.subheader(f"🔮 Churn Prediction: {pred}")
    st.write(f"**Probability of churn:** {prob:.2f}")

    # SHAP explanation
    explainer = shap.Explainer(model)
    shap_values = explainer(sample)

    st.subheader("🔍 SHAP Waterfall Plot")
    plt.figure()  # Start a new matplotlib figure
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    fig = plt.gcf()
    st.pyplot(fig)

# Option 2: Upload your own CSV
uploaded = st.file_uploader("📁 Or upload a CSV file", type="csv")

if uploaded is not None:
    user_data = pd.read_csv(uploaded)

    # Ensure same columns
    missing_cols = set(X.columns) - set(user_data.columns)
    if missing_cols:
        st.error(f"Uploaded data is missing columns: {missing_cols}")
    else:
        st.write("### Uploaded Data:")
        st.dataframe(user_data)

        # Use only first row for prediction
        row = user_data.iloc[[0]]
        prob = model.predict_proba(row)[0][1]
        pred = "Yes" if prob >= 0.5 else "No"

        st.subheader(f"🔮 Churn Prediction: {pred}")
        st.write(f"**Probability of churn:** {prob:.2f}")

        # SHAP plot for uploaded row
        explainer = shap.Explainer(model)
        shap_values = explainer(row)

        st.subheader("🔍 SHAP Waterfall Plot")
        plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        fig = plt.gcf()
        st.pyplot(fig)
