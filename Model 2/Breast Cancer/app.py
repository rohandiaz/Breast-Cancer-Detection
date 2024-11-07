import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(r'F:\Breast_Cancer_Prediction\Model 2\Breast Cancer\breast_cancer_model.pkl')

# Title of the app
st.title("Breast Cancer Prediction App")

# Instructions
st.write("Please provide the following details to predict if there is a likelihood of breast cancer.")

# Collecting input data from the user
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0, value=23.0, step=0.1)

race = st.selectbox("Race", options=["Asian", "Black", "Hispanic", "White", "Other"])
family_history = st.selectbox("Family History of Breast Cancer", options=["Yes", "No"])
breast_density = st.selectbox("Breast Density", options=["High", "Medium", "Low"])
mammogram_results = st.selectbox("Prior Mammogram Results", options=["Normal", "Abnormal"])
lifestyle_factors = st.selectbox("Lifestyle Factors (e.g., smoking, alcohol consumption)", options=["Healthy", "Unhealthy"])
breast_pain = st.selectbox("Breast Pain", options=["Yes", "No"])
nipple_discharge = st.selectbox("Discharge from Nipple", options=["Yes", "No"])
lump_in_breast = st.selectbox("Lump in Breast", options=["Yes", "No"])

# Predict button
if st.button("Predict"):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Body Mass Index (BMI)': [bmi],
        'Race': [race],
        'Family History of Breast Cancer': [family_history],
        'Breast Density': [breast_density],
        'Prior Mammogram Results': [mammogram_results],
        'Lifestyle Factors (e.g., smoking, alcohol consumption)': [lifestyle_factors],
        'Breast Pain': [breast_pain],
        'Discharge from Nipple': [nipple_discharge],
        'Lump in Breast': [lump_in_breast]
    })

    # Perform prediction
    prediction = model.predict(input_data)
    prediction_label = "Cancer" if prediction[0] else "No Cancer"

    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_data)
    cancer_probability = prediction_proba[0][1] * 100  # Probability for "Cancer" class
    no_cancer_probability = prediction_proba[0][0] * 100  # Probability for "No Cancer" class

    # Display result
    st.subheader("Prediction")
    st.write(f"The model predicts: **{prediction_label}**")
    
    # Display probabilities
    st.subheader("Prediction Probabilities")
    st.write(f"Probability of Cancer: **{cancer_probability:.2f}%**")
    st.write(f"Probability of No Cancer: **{no_cancer_probability:.2f}%**")
