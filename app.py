import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
threshold = joblib.load('optimal_threshold.pkl')
explainer = joblib.load('shap_explainer.pkl')
feature_names = joblib.load('feature_names.pkl')

age_map = {'18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, '40-44': 4, '45-49': 5,
           '50-54': 6, '55-59': 7, '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, '80 or older': 12}

gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}

diabetic_map = {'No': 0, 'No, borderline diabetes': 1, 'Yes (during pregnancy)': 2, 'Yes': 3}

race_map = {'Asian': 'Race_Asian', 'Black': 'Race_Black', 'Hispanic': 'Race_Hispanic', 
            'Other': 'Race_Other', 'White': 'Race_White'}

binary_map = {'No': 0, 'Yes': 1}

def create_feature_vector(user_inputs):
    
    # Initialize feature vector with zeros
    features = np.zeros(len(feature_names))
    feature_dict = {}
    
    feature_dict['BMI'] = user_inputs['BMI']
    feature_dict['PhysicalHealth'] = user_inputs['PhysicalHealth']
    feature_dict['MentalHealth'] = user_inputs['MentalHealth']
    feature_dict['GenHealth_Encoded'] = gen_health_map[user_inputs['GenHealth']]
    
    # Map race (one-hot encoded)
    race_feature = race_map[user_inputs['Race']]
    feature_dict[race_feature] = 1
    
    # Map binary features
    feature_dict['Smoking_yes'] = binary_map[user_inputs['Smoking']]
    feature_dict['AlcoholDrinking_Yes'] = binary_map[user_inputs['AlcoholDrinking']]
    feature_dict['Stroke_Yes'] = binary_map[user_inputs['Stroke']]
    feature_dict['DiffWalking_Yes'] = binary_map[user_inputs['DiffWalking']]
    feature_dict['Sex_Male'] = 1 if user_inputs['Sex'] == 'Male' else 0
    feature_dict['PhysicalActivity_Yes'] = binary_map[user_inputs['PhysicalActivity']]
    feature_dict['Asthma_Yes'] = binary_map[user_inputs['Asthma']]
    feature_dict['KidneyDisease_Yes'] = binary_map[user_inputs['KidneyDisease']]
    feature_dict['SkinCancer_Yes'] = binary_map[user_inputs['SkinCancer']]
    
    # Map age category (one-hot encoded)
    age_category = user_inputs['AgeCategory']
    if age_category != '18-24':  # 18-24 is the reference category (all zeros)
        age_feature = f'AgeCategory_{age_category}'
        if age_feature in feature_names:
            feature_dict[age_feature] = 1
    
    # Map diabetic status (one-hot encoded, but only 'Yes' is in features)
    if user_inputs['Diabetic'] == 'Yes':
        feature_dict['Diabetic_Yes'] = 1
    
    # Fill the feature vector
    for i, feature_name in enumerate(feature_names):
        if feature_name in feature_dict:
            features[i] = feature_dict[feature_name]
    
    return features.reshape(1, -1)

st.title("Heart Disease Risk Predictor")

st.header("Enter Your Health Information")

col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    physical_health = st.slider("Physical Health (days not good in past 30 days)", 0, 30, 0)
    mental_health = st.slider("Mental Health (days not good in past 30 days)", 0, 30, 0)
    
    gen_health = st.selectbox("General Health", list(gen_health_map.keys()))
    age_category = st.selectbox("Age Category", list(age_map.keys()))
    race = st.selectbox("Race", list(race_map.keys()))
    sex = st.selectbox("Sex", ["Male", "Female"])
    diabetic = st.selectbox("Diabetic", list(diabetic_map.keys()))

with col2:
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    alcohol_drinking = st.selectbox("Alcohol Drinking", ["No", "Yes"])
    stroke = st.selectbox("Had Stroke", ["No", "Yes"])
    diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"])
    physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
    asthma = st.selectbox("Asthma", ["No", "Yes"])
    kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"])
    skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"])

if st.button("Predict Heart Disease Risk", type="primary"):
    user_inputs = {
        'BMI': bmi,
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'GenHealth': gen_health,
        'Race': race,
        'Smoking': smoking,
        'AlcoholDrinking': alcohol_drinking,
        'Stroke': stroke,
        'DiffWalking': diff_walking,
        'Sex': sex,
        'AgeCategory': age_category,
        'Diabetic': diabetic,
        'PhysicalActivity': physical_activity,
        'Asthma': asthma,
        'KidneyDisease': kidney_disease,
        'SkinCancer': skin_cancer
    }
    
    features = create_feature_vector(user_inputs)
    
    features_scaled = scaler.transform(features)
    
    prediction_proba = model.predict_proba(features_scaled)[0]
    prediction = (prediction_proba[1] > threshold).astype(int)
    
    st.header("Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ HIGH RISK: Heart disease detected")
        else:
            st.success("✅ LOW RISK: No heart disease detected")
    
    with col2:
        st.metric("Risk Probability", f"{prediction_proba[1]:.2%}")
        st.metric("Optimal Threshold", f"{threshold:.3f}")
    
    # Risk interpretation
    st.subheader("Risk Interpretation")
    risk_percent = prediction_proba[1] * 100
    
    if risk_percent < 20:
        st.info(f"Your risk is {risk_percent:.1f}% - This is considered low risk.")
    elif risk_percent < 50:
        st.warning(f"Your risk is {risk_percent:.1f}% - This is considered moderate risk.")
    else:
        st.error(f"Your risk is {risk_percent:.1f}% - This is considered high risk.")
    
    st.markdown("---")
    st.caption("⚠️ This prediction is for educational purposes only. Please consult with a healthcare professional for medical advice.")