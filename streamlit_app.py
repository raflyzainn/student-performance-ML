import streamlit as st
import pandas as pd

st.title(':mortar_board: Student Performance in Exams by Machine Learning')

st.info('📊 Student Performance Analysis by Machine Learning')

df = pd.read_csv("StudentsPerformance.csv")

df.columns = [str(col).strip() for col in df.columns]

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# Melihat isi dari dataset 
with st.expander('Data'):
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


with st.expander('🧹 Pre-Processing Data'):
    st.markdown("Klik tombol berikut untuk melakukan transformasi:")

    # Tombol drop missing values
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.success("✅ Missing values telah dihapus.")
        st.dataframe(df)

    # Tombol encode kategorikal
    if st.button("🔠 Encode Kolom Kategorikal"):
        categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        df = pd.get_dummies(df, columns=categorical_cols)
        st.success("✅ Encoding berhasil dilakukan.")
        st.dataframe(df)

  



