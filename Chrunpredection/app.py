import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the app
st.title("🌟 Customer Retention Prediction App 🌟")
st.markdown("**Use this app to predict whether a customer is likely to stay or leave based on their details.**")

def user_input_features():
    # Pre-filled values likely to result in a churn prediction (for demonstration purposes)
    CreditScore = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, step=1, value=350)
    Geography = st.sidebar.selectbox("Geography", ("France", "Spain", "Germany"), index=2)
    Gender = st.sidebar.selectbox("Gender", ("Male", "Female"), index=1)
    Age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1, value=55)
    Tenure = st.sidebar.number_input("Tenure (years)", min_value=0, max_value=10, step=1, value=1)
    Balance = st.sidebar.number_input("Balance", min_value=0.0, step=100.0, value=5000.0)
    NumOfProducts = st.sidebar.number_input("Number of Products", min_value=1, max_value=4, step=1, value=1)
    HasCrCard = st.sidebar.selectbox("Has Credit Card?", (0, 1), index=1)
    IsActiveMember = st.sidebar.selectbox("Is Active Member?", (0, 1), index=0)
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=0.0, step=1000.0, value=30000.0)

    # One-hot encode Geography
    geography_dummies = [0, 0]  # Default for France
    if Geography == "Spain":
        geography_dummies = [1, 0]
    elif Geography == "Germany":
        geography_dummies = [0, 1]

    # Encode Gender
    gender_dummy = 1 if Gender == "Male" else 0

    # Combine all features into a single array
    features = np.array([
        CreditScore, Age, Tenure, Balance, NumOfProducts,
        HasCrCard, IsActiveMember, EstimatedSalary, *geography_dummies, gender_dummy
    ], dtype=float)

    return features

# Collect user input
user_data = user_input_features()

# Add a button to trigger the prediction
if st.button("🔍 Predict Customer Retention"):
    with st.spinner("Analyzing the data..."):
        # Ensure the input shape matches the model's expectation
        assert user_data.shape[0] == 11, "Input data must have exactly 11 features."

        # Reshape the data for the model
        user_data = user_data.reshape(1, -1)

        # Standardize features using the loaded scaler
        scaled_data = scaler.transform(user_data)

        # Predict churn class (0 or 1)
        prediction = model.predict(scaled_data)[0]  # Extract the first (and only) prediction

        # Add custom CSS for styling
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
            .positive {
                background-color: #4CAF50; /* Green */
            }
            .negative {
                background-color: #F44336; /* Red */
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Display conversational prediction result with styling
        st.subheader("Prediction")
        if prediction == 1:
            st.markdown(
                """
                <div class="result-card negative">
                Based on the information provided, it seems this customer might be considering leaving. 
                You may want to take proactive steps to improve their experience.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="result-card positive">
                This customer appears satisfied and likely to continue with your services. 
                Keep maintaining the good relationship!
                </div>
                """,
                unsafe_allow_html=True
            )
