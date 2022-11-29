import streamlit as st
import Dataset


st.subheader("Prediksi harga hp")

battery_power = st.number_input("Kapasitas baterai",min_value=501,max_value=1998)
clock_speed = st.number_input("Kecepatan processor (dalam GHz)",min_value=0.5,max_value=3.0)
fc = st.number_input("Kamera Depan",min_value=0,max_value=19)
int_memory = st.number_input("Memory Internal",min_value=2,max_value=64)
n_cores = st.number_input("Jumlah core processor",min_value=1,max_value=8)
pc = st.number_input("Kamera Belakang",min_value=0,max_value=20)
ram = st.number_input("Ram (dalam Mb)",min_value=256,max_value=3998)
sc_h = st.number_input("Tinggi Layar",min_value=5,max_value=19)
sc_w = st.number_input("Lebar Layar",min_value=0,max_value=18)

columns = st.columns((2, 0.6, 2))
save = columns[1].button("Submit")

if save:
    # normalisasi data
    data = Dataset.normalisasi([battery_power,clock_speed,fc,int_memory,n_cores,pc,ram,sc_h,sc_w])
    # prediksi data
    prediksi = Dataset.ranforest(data)
    # cek prediksi

    with st.spinner("Tunggu Sebentar Masih Proses..."):
        st.write("Golongan ",prediksi[-1])  
        if prediksi[-1]==0:
            st.subheader("Hp anda termasuk hp dengan harga murah")
        if prediksi[-1]==1:
            st.subheader("Hp anda termasuk hp dengan harga sedang")
        if prediksi[-1]==2:
            st.subheader("Hp anda termasuk hp dengan harga mahal")
        if prediksi[-1]==3:
            st.subheader("Hp anda termasuk hp dengan harga sangat mahal")

   


