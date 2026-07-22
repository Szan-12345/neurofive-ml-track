import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the saved model and columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🚀 Machine Learning Web App - Customer Churn Predictor")
st.write(
    "Enter customer details below to predict churn probability in real time!"
)

# Create input widgets for key features
tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.number_input(
    "Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0
)
total_charges = st.number_input(
    "Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0
)
contract_type = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

# Prediction Button
if st.button("Predict Churn"):
  # Create a blank input dataframe matching original training structure
  input_data = pd.DataFrame(0, index=[0], columns=model_columns)

  # Assign user inputs to corresponding columns safely
  if "tenure" in input_data.columns:
    input_data["tenure"] = tenure
  if "MonthlyCharges" in input_data.columns:
    input_data["MonthlyCharges"] = monthly_charges
  if "TotalCharges" in input_data.columns:
    input_data["TotalCharges"] = total_charges

  # Handle dummy variables manually based on selections
  if contract_type == "One year" and "Contract_One year" in input_data.columns:
    input_data["Contract_One year"] = 1
  elif (
      contract_type == "Two year" and "Contract_Two year" in input_data.columns
  ):
    input_data["Contract_Two year"] = 1

  if (
      internet_service == "Fiber optic"
      and "InternetService_Fiber optic" in input_data.columns
  ):
    input_data["InternetService_Fiber optic"] = 1
  elif internet_service == "No" and "InternetService_No" in input_data.columns:
    input_data["InternetService_No"] = 1

  # Make prediction
  prediction = model.predict(input_data)
  prediction_proba = model.predict_proba(input_data)

  if prediction[0] == 1:
    st.error(
        f"⚠️ High Risk: This customer is likely to churn! (Probability:"
        f" {prediction_proba[0][1]*100:.2f}%)"
    )
  else:
    st.success(
        f"✅ Low Risk: This customer is likely to stay. (Probability of"
        f" Churn: {prediction_proba[0][1]*100:.2f}%)"
    )
