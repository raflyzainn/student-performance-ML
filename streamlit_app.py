import streamlit as st
import pandas as pd

st.title(':mortar_board: Student Performance in Exams by Machine Learning')

st.info('📊 Student Performance Analysis by Machine Learning')

df.columns = [str(col).strip() for col in df.columns]

df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("StudentsPerformance.csv")
  df

  X = df.drop(columns=['average_score'])
  st.write("🧩 **Features (X)**")
  st.dataframe(X)

  y = df['average_score']
  st.write("🎯 **Target (y)**")
  st.dataframe(y)
