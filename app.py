import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")
st.write("Passenger information input form:")

# Input UI elements
pclass = st.selectbox("Pclass (Passenger Class)", [1, 2, 3], index=2)
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", value=22.0)
fare = st.number_input("Fare", value=7.25)
embarked = st.selectbox("Embarked Port", ["S", "C", "Q"])
family_size = st.number_input("Family Size", value=1)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'Pclass': int(pclass),
        'Sex': sex,
        'Age': float(age),
        'Fare': float(fare),
        'Embarked': embarked,
        'FamilySize': int(family_size)
    }])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("Result: Survived 🎉")
    else:
        st.error("Result: Did Not Survive ❌")