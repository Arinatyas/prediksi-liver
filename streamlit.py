import joblib
import numpy as np

model = joblib.load('random_forest_model.pkl')

def predict_liver_disease():
    try:
        age = float(input("Masukkan umur (Age): "))
        gender = int(input("Masukkan jenis kelamin (Gender, 0 = Female, 1 = Male): "))
        total_bilirubin = float(input("Masukkan Total Bilirubin: "))
        direct_bilirubin = float(input("Masukkan Direct Bilirubin: "))
        alkaline_phosphotase = float(input("Masukkan Alkaline Phosphotase: "))
        alamine_aminotransferase = float(input("Masukkan Alamine Aminotransferase: "))
        aspartate_aminotransferase = float(input("Masukkan Aspartate Aminotransferase: "))
        total_protiens = float(input("Masukkan Total Proteins: "))
        albumin = float(input("Masukkan Albumin: "))
        albumin_and_globulin_ratio = float(input("Masukkan Albumin and Globulin Ratio: "))

        input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                               alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                               albumin, albumin_and_globulin_ratio]])
        prediction = model.predict(input_data)

        result = "Positive" if prediction[0] == 1 else "Negative"
        print(f"The result is: {result}")

    except ValueError:
        print("Invalid input. Please enter valid numbers.")

if __name__ == "__main__":
    predict_liver_disease()

import joblib
import numpy as np
import streamlit as st

# Load the model
model = joblib.load('random_forest_model.pkl')

def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                          alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                          albumin, albumin_and_globulin_ratio):
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                            albumin, albumin_and_globulin_ratio]])
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# Streamlit app
st.title('Liver Disease Prediction')

age = st.number_input('Masukkan umur (Age)')
gender = st.radio('Masukkan jenis kelamin (Gender)', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
total_bilirubin = st.number_input('Masukkan Total Bilirubin')
direct_bilirubin = st.number_input('Masukkan Direct Bilirubin')
alkaline_phosphotase = st.number_input('Masukkan Alkaline Phosphotase')
alamine_aminotransferase = st.number_input('Masukkan Alamine Aminotransferase')
aspartate_aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase')
total_protiens = st.number_input('Masukkan Total Proteins')
albumin = st.number_input('Masukkan Albumin')
albumin_and_globulin_ratio = st.number_input('Masukkan Albumin and Globulin Ratio')

if st.button('Predict'):
    result = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                   alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                                   albumin, albumin_and_globulin_ratio)
    st.write(f'The result is: {result}')
