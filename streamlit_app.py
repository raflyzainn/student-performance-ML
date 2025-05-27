import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

  # Visualisasi: Bar Chart
  st.subheader("🍽️ Rata-Rata Skor Berdasarkan Jenis Lunch")
  avg_by_lunch = df.groupby("lunch")[["average_score"]].mean()
  
  # Plot
  fig, ax = plt.subplots()
  avg_by_lunch.plot(kind='bar', ax=ax, color=['#1E90FF'])  # biru
  ax.set_ylabel("Average Score")
  ax.set_xlabel("Lunch Type")
  ax.set_title("Rata-Rata Skor per Lunch Type")
  st.pyplot(fig)

# Melihat visualisasi data
with st.expander('📊 Data Visualization'):
  # Menghitung rata-rata per gender
  avg_by_gender = df.groupby("gender")[["average_score"]].mean()
  st.subheader("📊 Rata-rata Skor per Gender")
  st.bar_chart(avg_by_gender)

  st.subheader("📊 Rata-rata Skor per Kategori (contoh: gender)")

  if "gender" in df.columns:
    avg_by_cat = df.groupby("lunch")[["average_score"]].mean().sort_values(by="average_score", ascending=False)
    st.bar_chart(avg_by_cat)


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

  



