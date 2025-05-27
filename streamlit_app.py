import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.title(':mortar_board: Student Performance in Exams by Machine Learning')

st.info('📊 Student Performance Analysis by Machine Learning')

df = pd.read_csv("StudentsPerformance.csv")

df.columns = [str(col).strip() for col in df.columns]

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Melihat isi dari dataset 
with st.expander('📁 Data'):
  st.write('**Raw Data**')
  df
  
  # Menampilkan fitur - fitur yang tersedia
  X = df.drop(columns=['average_score'])
  st.write("🧩 **Features (X)**")
  st.dataframe(X)
  
  # Menampilkan target variable yaitu nilai rata - rata
  y = df['average_score']
  st.write("🎯 **Target (y)**")
  st.dataframe(y)

# Melihat visualisasi data
with st.expander('📊 Data Visualization'):
  # Menghitung rata-rata per gender
  avg_by_gender = df.groupby("gender")[["average_score"]].mean()
  st.subheader("📊 Rata-rata Skor per Gender")
  st.bar_chart(avg_by_gender)

  st.subheader("🍽️ Rata-Rata Skor Berdasarkan Jenis Lunch")

  if "gender" in df.columns:
    avg_by_cat = df.groupby("lunch")[["average_score"]].mean().sort_values(by="average_score", ascending=False)
    st.bar_chart(avg_by_cat)


with st.expander('🧹 Pre-Processing Data'):
    if st.button("Drop Missing Values"):
        df.dropna(inplace=True)
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding Kolom Kategorikal"):
        le = LabelEncoder()

        # Encode kolom yang kamu anggap penting (bisa disesuaikan)
        df['gender'] = le.fit_transform(df['gender'])  # female=0, male=1
        df['lunch'] = le.fit_transform(df['lunch'])    # free/reduced=0, standard=1
        df['test preparation course'] = le.fit_transform(df['test preparation course'])

        # OPTIONAL: encode tambahan
        df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
        df['parental level of education'] = le.fit_transform(df['parental level of education'])

        st.success("✅ Label Encoding selesai.")
        st.dataframe(df)

    if st.button("🔄 Pisahkan Fitur (X) dan Target (y)"):
        X = df.drop(columns=['average_score'])
        y = df['average_score']
        st.write("🧩 X (fitur):")
        st.dataframe(X)
        st.write("🎯 y (target):")
        st.dataframe(y)

    # Tombol encode kategorikal
    if st.button("🔠 Encode Kolom Kategorikal"):
        categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        df = pd.get_dummies(df, columns=categorical_cols)
        st.success("✅ Encoding berhasil dilakukan.")
        st.dataframe(df)

  



