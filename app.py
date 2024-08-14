import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('stroke_prediction_model.pkl')

# Title of the web app
st.title('Stroke Prediction App')

# Input features
st.sidebar.header('Input Features')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    ever_married = st.sidebar.selectbox('Ever Married', ['Yes', 'No'])
    work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    Residence_type = st.sidebar.selectbox('Residence Type', ['Urban', 'Rural'])
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', 0.0, 300.0, 100.0)
    bmi = st.sidebar.slider('BMI', 0.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }
    features = pd.DataFrame(data, index=[0])
    
    # One-hot encode the categorical variables (same as during model training)
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    features = pd.get_dummies(features, columns=categorical_columns, drop_first=True)
    
    # Ensure the input data has the same columns as the training data
    # If the training data had additional columns (due to drop_first=True), add missing columns with a value of 0
    missing_cols = set(model_columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0

    features = features[model_columns]
    
    return features

# Load the model columns (you should save the columns during model training and load them here)
model_columns = joblib.load('model_columns.pkl')

input_df = user_input_features()

# Make predictions
prediction_proba = model.predict_proba(input_df)[:, 1]
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write('Probability of stroke: ', prediction_proba[0])
st.write('Predicted class: ', 'Stroke' if prediction[0] == 1 else 'No Stroke')

