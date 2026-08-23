import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("fraud_scaler.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown(
    "### Machine Learning based transaction classification"
)

st.divider()

# -----------------------------
# Information
# -----------------------------
st.subheader("🔍 Transaction Analysis")

st.info(
    "Enter the transaction details below. "
    "The trained Machine Learning model will classify the transaction "
    "as Genuine or Fraudulent."
)

# -----------------------------
# Amount Input
# -----------------------------
amount = st.number_input(
    "💰 Transaction Amount",
    min_value=0.0,
    value=100.0,
    step=1.0
)

# -----------------------------
# Feature Inputs
# -----------------------------
st.subheader("📊 Transaction Features")

features = []

cols = st.columns(4)

for i in range(28):
    with cols[i % 4]:
        value = st.number_input(
            f"V{i+1}",
            value=0.0,
            format="%.6f"
        )
        features.append(value)

# -----------------------------
# Time Input
# -----------------------------
time_value = st.number_input(
    "⏱️ Transaction Time",
    min_value=0.0,
    value=0.0
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔎 Analyze Transaction", use_container_width=True):

    # Arrange features in correct order:
    # Time + V1-V28 + Amount
    transaction = [time_value] + features + [amount]

    transaction_df = pd.DataFrame(
        [transaction],
        columns=[
            "Time",
            "V1", "V2", "V3", "V4", "V5",
            "V6", "V7", "V8", "V9", "V10",
            "V11", "V12", "V13", "V14", "V15",
            "V16", "V17", "V18", "V19", "V20",
            "V21", "V22", "V23", "V24", "V25",
            "V26", "V27", "V28",
            "Amount"
        ]
    )

    # Scale transaction
    transaction_scaled = scaler.transform(transaction_df)

    # Prediction
    prediction = model.predict(transaction_scaled)[0]

    probability = model.predict_proba(transaction_scaled)[0]

    # -----------------------------
    # Result
    # -----------------------------
    st.divider()
    st.subheader("📋 Prediction Result")

    if prediction == 1:

        st.error("🚨 FRAUDULENT TRANSACTION")

        st.write(
            f"Fraud Probability: **{probability[1] * 100:.2f}%**"
        )

    else:

        st.success("✅ GENUINE TRANSACTION")

        st.write(
            f"Genuine Probability: **{probability[0] * 100:.2f}%**"
        )

    # -----------------------------
    # Probability Chart
    # -----------------------------
    result_df = pd.DataFrame(
        {
            "Transaction Type": ["Genuine", "Fraud"],
            "Probability": [
                probability[0],
                probability[1]
            ]
        }
    )

    st.subheader("📈 Prediction Probability")

    st.bar_chart(
        result_df.set_index("Transaction Type")
    )

# -----------------------------
# Model Information
# -----------------------------
st.divider()

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Logistic Regression")

with col2:
    st.metric("Fraud Recall", "91.84%")

with col3:
    st.metric("Dataset Transactions", "284,807")

st.caption(
    "Credit Card Fraud Detection | Machine Learning Project"
)
