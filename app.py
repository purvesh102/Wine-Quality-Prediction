import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Red wine prediction system")

col1, col2, col3 = st.beta_columns(3)

fixed_acidity = np.float(col1.text_input("Fixed Acidity",7.4))
volatile_acidity = np.float(col2.text_input("Volatile Acidity",0.7))
citric_acidity = np.float(col3.text_input("Citric Acid",0))
residual_sugar = np.float(col1.text_input("Residual Sugar",1.9))
chlorides = np.float(col2.text_input("Chlorides",0.076))
free_sulphur_dioxide =np.float(col3.text_input("Free Sulphur dioxide",11))
total_sulphur_dioxide = np.float(col1.text_input("Total Sulphur dioxide",34))
density = np.float(col2.text_input("Density",0.9978))
ph = np.float(col3.text_input("PH",3.51))
sulphates = np.float(col1.text_input("Sulphate",0.56))
alcohol = np.float(col2.text_input("Alcohol",0.7))

features_num = ['fixed acidity','volatile acidity','citric acid',
        'residual sugar','chlorides','free sulfur dioxide',
        'total sulfur dioxide','density','pH','sulphate','alcohol']



sample = [fixed_acidity,
          volatile_acidity,
          citric_acidity,
          residual_sugar,
          chlorides,
          free_sulphur_dioxide,
          total_sulphur_dioxide,
          density,
          ph,
          sulphates,
          alcohol
          ]

sample_df = pd.DataFrame([sample],columns = features_num)
model = pickle.load(open('model.pkl',"rb"))


if st.button('Predict'):
    result = model.predict(sample_df)
    if result == 3:
        st.header("Lowest Quality")
    elif result == 4:
        st.header("Low Quality")
    elif result == 5:
        st.header("Medium Quality")
    elif result == 6:
        st.header("Average Quality")
    elif result == 7:
        st.header("Good Quality")
    else:
        st.header("Best Quality")