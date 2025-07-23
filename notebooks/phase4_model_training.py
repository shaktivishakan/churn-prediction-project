import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load data
df = pd.read_csv("data/processed_telco.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Step 1    Train-test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 2    Logistic Regression

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
lr.preds = lr.predict(X_test)

print("\n Logistic Regression Report")
print(classification_report(y_test, lr.preds))
print("AUC:",roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]))

# Save Model
joblib.dump(lr, "models/lr_model.joblib")

# Step 3    Random Forest

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.preds = rf.predict(X_test)

print("\n Random Forest Report")
print(classification_report(y_test, rf.preds))
print("AUC:",roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# Save Model
joblib.dump(rf, "models/rf_model.joblib")

# Step 4    XGBoost Classifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.preds = xgb.predict(X_test)

print("\n XGBoost Report")
print(classification_report(y_test, xgb.preds))
print("AUC:",roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))

# Save Model
joblib.dump(xgb, "models/xgb_model.joblib")
