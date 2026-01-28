# ---------------- app1.py ----------------
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Telco Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction (XGBoost)")

# ---------------- Load Dataset ----------------
DATA_PATH = r"C:\Users\kasar\Downloads\telco1.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Target mapping
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Fix TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # Drop ID column if exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)

    return df

df = load_data()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------- Features & Target ----------------
X = df.drop('Churn', axis=1)
y = df['Churn']

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- XGBoost Model ----------------
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = model.predict(X_test)

# ---------------- Metrics ----------------
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.2f}")

st.write("### Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

# ---------------- Classification Report Table ----------------
st.subheader("📋 Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df)

# ---------------- Feature Importance ----------------
st.subheader("⭐ Top 20 Important Features")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(20),
    ax=ax
)
ax.set_title("Top 20 Feature Importances (XGBoost)")
st.pyplot(fig)

# ---------------- Single Prediction ----------------
st.subheader("🔮 Predict Churn for a New Customer")

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 20, 120, 70)
total_charges = st.slider("Total Charges", 0, 9000, 1500)

input_df = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Add missing columns
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN ({probability:.2%})")
    else:
        st.success(f"✅ Customer is likely to STAY ({probability:.2%})")

st.info(
    "This app uses **XGBoost**, an advanced **ensemble boosting algorithm**, "
    "commonly used in real-world ML competitions and industry churn systems."
)
