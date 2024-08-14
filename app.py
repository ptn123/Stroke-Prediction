import streamlit as st
import joblib
import pandas as pd

# Load the model and model columns
model = joblib.load('stroke_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Streamlit app title
st.title('Stroke Prediction')

# Inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0)
hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
bmi = st.number_input('BMI', min_value=0.0)
smoking_status = st.selectbox('Smoking Status', ['Never smoked', 'Formerly smoked', 'Smokes'])

# Convert inputs to DataFrame
input_data = pd.DataFrame({
    'gender': [gender],
    'age': [age],
    'hypertension': [1 if hypertension == 'Yes' else 0],
    'heart_disease': [1 if heart_disease == 'Yes' else 0],
    'ever_married': [1 if ever_married == 'Yes' else 0],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'smoking_status': [smoking_status]
})

# Encode categorical variables
input_data_encoded = pd.get_dummies(input_data, columns=['gender', 'work_type', 'Residence_type', 'smoking_status'])

# Ensure all necessary columns are present
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button('Predict'):
    # Predict
    prediction_proba = model.predict_proba(input_data_encoded)[:, 1]
    st.write(f'Probability of stroke: {prediction_proba[0]:.2f}')

    # Provide interpretation
    if prediction_proba[0] > 0.5:
        st.write("The model predicts that there is a high likelihood of stroke.")
    else:
        st.write("The model predicts that there is a low likelihood of stroke.")
