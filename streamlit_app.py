import streamlit as st
import pandas as pd

st.title(':mortar_board: Student Performance in Exams by Machine Learning')

st.info('📊 Student Performance Analysis by Machine Learning')

df = pd.read_csv("StudentsPerformance.csv")

df.columns = [str(col).strip() for col in df.columns]

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

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

  # Menghitung rata-rata per gender
  avg_by_gender = df.groupby("gender")[["average_score"]].mean()
  st.subheader("📊 Rata-rata Skor per Gender")
  st.area_chart(avg_by_gender)
