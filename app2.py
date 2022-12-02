import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Stroke Prediction App
This app predicts If a patient has a Stroke
Data obtained from Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset.
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    gender  = st.sidebar.selectbox('Gender',(0,1,2))
    hypertension = st.sidebar.selectbox('Hypertension',(0,1))
    heart_disease = st.sidebar.selectbox('Heart Disease ',(0,1))
    ever_married = st.sidebar.selectbox('Maried: '(0,1))
    work_type = st.sidebar.selectbox('Work Type',("Never","Govt_job","Private","Childern","employed"))
    residence_type = st.sidebar.selectbox('Resident Type: ',("Urban","Rural"))
    smoking_status = st.sidebar.selectbox('Smoking Status',("formerl","Unknown","never","smoking"))

    data = {'age': age,
            'gender': gender, 
            'hypertension': hypertension,
            'ever_married':ever_married,
            'work_type': work_type,
            'residence_type': residence_type,
            'smoking_status': smoking_status,
        
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('healthcare.csv')
heart_dataset = heart_dataset.drop(columns=['stroke'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
df = pd.get_dummies(df, columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level',
       'smoking_status',])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)