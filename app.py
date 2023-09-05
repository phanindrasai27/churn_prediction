import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Load the trained XGBoost model from the pickle file
with open("random_forest_model.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)

# Define a function for making predictions
def predict_churn(input_data):
    # One-hot encode 'Gender' and 'Location' columns
    input_data = pd.get_dummies(input_data, columns=["Gender", "Location"])
    
    # Make predictions using the loaded XGBoost model
    predictions = xgb_model.predict(input_data)
    
    # Return the predicted churn values
    return predictions

# Create a Streamlit app
st.title("Customer Churn Prediction")

# Input fields for customer data
age = st.number_input("Age", min_value=0, max_value=100, step=1)
subscription_length = st.number_input("Subscription Length (Months)", min_value=0, max_value=100, step=1)
monthly_bill = st.number_input("Monthly Bill", min_value=0.0, step=1.0)
total_usage_gb = st.number_input("Total Usage (GB)", min_value=0, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
location = st.selectbox("Location", ["Houston", "Los Angeles", "Miami", "New York"])

# Create a dictionary with the input data
input_data = {
    "Age": [age],
    "Subscription_Length_Months": [subscription_length],
    "Monthly_Bill": [monthly_bill],
    "Total_Usage_GB": [total_usage_gb],
    "Gender": [gender],
    "Location": [location]
}

# Make predictions when the user clicks the "Predict" button
if st.button("Predict"):
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame(input_data)
    
    # Call the predict_churn function with the input data
    predictions = predict_churn(input_df)
    
    # Display the prediction result
    if predictions[0] == 0:
        st.write("Prediction: Not Churned")
    else:
        st.write("Prediction: Churned")
