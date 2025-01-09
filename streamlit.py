import joblib
import numpy as np
import streamlit as st
import pandas as pd

# ✅ Fungsi untuk memuat model dengan caching
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

# ✅ Muat model
model = load_model()

# ✅ Fungsi untuk melakukan prediksi
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                          alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                          albumin, albumin_and_globulin_ratio):
    # Buat input data sebagai array 2D
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                            albumin, albumin_and_globulin_ratio]])
    # Lakukan prediksi
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# ✅ Aplikasi Streamlit
st.title('Liver Disease Prediction')

# 🔹 Input Data dari Pengguna
st.write("Masukkan data pasien untuk melakukan prediksi:")

age = st.number_input('Masukkan Umur (Age)', min_value=0, value=30)
gender = st.radio('Masukkan Jenis Kelamin (Gender)', [0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
total_bilirubin = st.number_input('Masukkan Total Bilirubin', min_value=0.0, value=1.0)
direct_bilirubin = st.number_input('Masukkan Direct Bilirubin', min_value=0.0, value=0.5)
alkaline_phosphotase = st.number_input('Masukkan Alkaline Phosphotase', min_value=0, value=150)
alamine_aminotransferase = st.number_input('Masukkan Alamine Aminotransferase', min_value=0, value=30)
aspartate_aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase', min_value=0, value=30)
total_protiens = st.number_input('Masukkan Total Proteins', min_value=0.0, value=6.0)
albumin = st.number_input('Masukkan Albumin', min_value=0.0, value=3.5)
albumin_and_globulin_ratio = st.number_input('Masukkan Albumin and Globulin Ratio', min_value=0.0, value=1.0)

# 🔹 Tombol untuk Prediksi
if st.button('Predict'):
    result = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                   alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                                   albumin, albumin_and_globulin_ratio)
    st.write(f'**Hasil Prediksi:** {result}')
