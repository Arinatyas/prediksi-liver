import joblib

joblib.dump(rf, 'random_forest_model.pkl')
print("Model berhasil disimpan ke 'random_forest_model.pkl'")
model = joblib.load('random_forest_model.pkl')
import joblib

joblib.dump(rf, 'random_forest_model.pkl')

import cloudpickle
import streamlit as st

# Simpan model menggunakan cloudpickle
with open('random_forest_model.pkl', 'wb') as f:
    cloudpickle.dump(rf, f)

@st.cache_resource
def load_model():
    with open('random_forest_model.pkl', 'rb') as f:
        return cloudpickle.load(f)

# ✅ Prediction function using the loaded model
def predict_liver_disease(input_data):
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# Memuat model hanya sekali
model = load_model()

# ✅ Fungsi untuk memprediksi penyakit hati
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                          alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                          albumin, albumin_and_globulin_ratio):
    input_data = np.array([[age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                            alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                            albumin, albumin_and_globulin_ratio]])
    prediction = model.predict(input_data)
    return "Positive" if prediction[0] == 1 else "Negative"

# ✅ Aplikasi Streamlit
st.title('Prediksi Penyakit Hati')

# 🔹 Inputan dari pengguna
age = st.number_input('Masukkan umur (Age)', min_value=0)
gender = st.radio('Masukkan jenis kelamin (Gender)', [0, 1], format_func=lambda x: 'Perempuan' if x == 0 else 'Laki-laki')
total_bilirubin = st.number_input('Masukkan Total Bilirubin')
direct_bilirubin = st.number_input('Masukkan Direct Bilirubin')
alkaline_phosphotase = st.number_input('Masukkan Alkaline Phosphotase')
alamine_aminotransferase = st.number_input('Masukkan Alamine Aminotransferase')
aspartate_aminotransferase = st.number_input('Masukkan Aspartate Aminotransferase')
total_protiens = st.number_input('Masukkan Total Proteins')
albumin = st.number_input('Masukkan Albumin')
albumin_and_globulin_ratio = st.number_input('Masukkan Albumin and Globulin Ratio')

# if st.button("Prediksi"):
#     result = predict_liver_disease(input_data)
#     st.write(f"**Hasil Prediksi:** {result}")
if st.button('Prediksi'):
    result = predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                                   alamine_aminotransferase, aspartate_aminotransferase, total_protiens,
                                   albumin, albumin_and_globulin_ratio)
    st.write(f'**Hasil Prediksi:** {result}')
