import streamlit as st
import numpy as np
from joblib import load

label = ['setosa', 'versicolor', 'virginica']

st.title('Iris Predictor')
choose_model = st.selectbox('Select model', ['SVM', 'LogisticRegression'])

if choose_model == 'SVM':
    model = load('./model/IRIS_SVM.joblib')
elif choose_model == 'LogisticRegression':
    model = load('./model/IRIS_LogisticRegression.joblib')
st.subheader('Iris features')
sl = st.slider('Sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sw = st.slider('Sepal width (cm)', min_value=0.0, max_value=10.0, step=0.1)
pl = st.slider('Petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
pw = st.slider('Petal width (cm)', min_value=0.0, max_value=10.0, step=0.1)
X = np.array([sl, sw, pl, pw]).reshape(1,-1)

hit = st.button('Predict')
if hit == True:
    st.subheader(label[model.predict(X)[0]])