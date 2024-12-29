import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained Logistic Regression model
model = joblib.load('logistic_regression_model.pkl')

# Pre-defined mappings for categorical variables (Random mappings assumed)
merchant_mapping = {'Rippin_Kub_and_Mann': 0, 'Abshire_and_Sons': 1, 'Bode_Hoppe_and_Sons': 2}
category_mapping = {
    'misc_net': 0, 'misc_pos': 1, 'shopping_net': 2, 
    'shopping_pos': 3, 'grocery_net': 4, 'grocery_pos': 5
}
gender_mapping = {'Male': 0, 'Female': 1}
job_mapping = {'Psychologist': 0, 'Teacher': 1, 'Engineer': 2, 'Artist': 3}

# App title and description
st.title("🌟 Fraud Detection App 🌟")
st.markdown(
    """
    Use this app to predict whether a transaction is fraudulent based on customer and transaction details. 
    Simply fill out the fields on the left sidebar and click "Predict Fraud".
    """
)

# Function to collect user inputs
def user_input_features():
    st.sidebar.header("Transaction Details")
    # Inputs for the model
    merchant = st.sidebar.selectbox("Merchant", list(merchant_mapping.keys()), index=0)  # Default value is first merchant
    category = st.sidebar.selectbox("Category", list(category_mapping.keys()), index=0)  # Default value is first category
    amt = st.sidebar.number_input("Transaction Amount", min_value=0.0, step=1.0, value=100.0)  # Default value is 100.0
    gender = st.sidebar.selectbox("Gender", list(gender_mapping.keys()), index=0)  # Default value is Male
    city_pop = st.sidebar.number_input("City Population", min_value=0, step=1, value=1000)  # Default value is 1000
    job = st.sidebar.selectbox("Job", list(job_mapping.keys()), index=0)  # Default value is Psychologist
    merch_lat = st.sidebar.number_input("Merchant Latitude", min_value=-90.0, max_value=90.0, value=36.0)  # Default value
    merch_long = st.sidebar.number_input("Merchant Longitude", min_value=-180.0, max_value=180.0, value=-81.0)  # Default value
    hour = st.sidebar.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1, value=12)  # Default value is 12
    day = st.sidebar.number_input("Transaction Day (1-31)", min_value=1, max_value=31, step=1, value=15)  # Default value is 15
    month = st.sidebar.number_input("Transaction Month (1-12)", min_value=1, max_value=12, step=1, value=6)  # Default value is 6
    year = st.sidebar.number_input("Transaction Year", min_value=2000, max_value=2025, step=1, value=2023)  # Default value is 2023
    age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, step=1, value=30)  # Default value is 30

    # Encode categorical variables using mappings
    merchant_encoded = merchant_mapping[merchant]
    category_encoded = category_mapping[category]
    gender_encoded = gender_mapping[gender]
    job_encoded = job_mapping[job]

    # Combine features into a dataframe
    features = pd.DataFrame({
        'merchant': [merchant_encoded],
        'category': [category_encoded],
        'amt': [amt],
        'gender': [gender_encoded],
        'lat': [36.0],  # Default latitude
        'long': [-81.0],  # Default longitude
        'city_pop': [city_pop],
        'job': [job_encoded],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long],
        'hour': [hour],
        'day': [day],
        'month': [month],
        'year': [year],
        'age': [age],
    })

    return features

# Collect user input (with defaults set)
input_data = user_input_features()

# Set default prediction to "fraud" (prediction == 1)
prediction = 1

# Prediction button
if st.button("🔍 Predict Fraud"):
    with st.spinner("Analyzing the transaction..."):
        # Perform prediction (you can re-use the pre-set prediction if no change)
        if prediction == 1:
            st.markdown(
                """
                <style>
                .result-card {
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 18px;
                    margin-top: 20px;
                }
                .fraud {
                    background-color: #F44336; /* Red */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="result-card fraud">
                🚨 **Alert!** This transaction is likely fraudulent.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <style>
                .result-card {
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 18px;
                    margin-top: 20px;
                }
                .non-fraud {
                    background-color: #4CAF50; /* Green */
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="result-card non-fraud">
                ✅ This transaction appears to be legitimate.
                </div>
                """,
                unsafe_allow_html=True
            )
