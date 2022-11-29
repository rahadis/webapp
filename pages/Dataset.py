import joblib
import pandas as pd
import streamlit as st

st.title("Studi Kasus Mobile Price Classification")
st.header("Link Dataset")
st.subheader("https://raw.githubusercontent.com/rahadis/datamining/main/train.csv")
st.header("Link Github")
st.subheader("https://github.com/rahadis")

st.header("Mobile Price Classification Dataset")
data = pd.read_csv('Pendat/mobileprice.csv')
data = data.drop(data.columns[0],axis=1)
df=pd.DataFrame(data)
st.dataframe(df)

st.subheader("Penjelasan")
st.write("battery_power = Kapasitas Baterai") 
st.write("clock_speed = Kecepatan Processor") 
st.write("fc = Kamera Depan") 
st.write("int_memory = Memori internal") 
st.write("n_cores = Jumlah Core GPU") 
st.write("pc = Kamera utama") 
st.write("ram = RAM") 
st.write("sc_h = Tinggi Layar") 
st.write("sc_w = Lebar Layar") 

def normalisasi(x):
    # import data test
    cols = ["battery_power","clock_speed","fc","int_memory","n_cores","pc","ram","sc_h","sc_w"]
    df = pd.DataFrame([x],columns=cols)
    data_test = pd.read_csv('Pendat/Datafix.csv')
    data_test = data_test.drop(data_test.columns[0],axis=1)
    # memasukkan data kedalam data test
    data_test = data_test.append(df,ignore_index = True)
    # return data_test yang sudah dinormalisasi
    return joblib.load('Pendat/Normalisasi.sav').fit_transform(data_test)

def knn(x):
    return joblib.load('Pendat/Model/KNN/KNNmodel29.pkl').predict(x)
def NB(x):
    return joblib.load('Pendat/Model/ModelNB.pkl').predict(x)
def ranforest(x):
    return joblib.load('Pendat/Model/randomforest.pkl').predict(x)