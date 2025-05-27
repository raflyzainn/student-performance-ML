import streamlit as st
import pandas as pd

st.title(':mortar_board: Student Performance in Exams by Machine Learning')

st.info('📊 Student Performance Analysis by Machine Learning')

with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv("StudentsPerformance.csv")
  df
