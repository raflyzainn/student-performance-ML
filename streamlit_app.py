import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

st.title(':mortar_board: Student Performance in Exams by Machine Learning')
st.info('📊 Student Performance Analysis by Machine Learning')

# ===== Inisialisasi data hanya sekali (saat awal atau refresh) =====
if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    st.session_state.df = df.copy()

# Gunakan df dari session_state
df = st.session_state.df

# ===== Data View =====
with st.expander('📁 Data'):
    st.write('**Raw Data**')
    st.dataframe(df)
    
    X = df.drop(columns=['average_score'])
    st.write("🧩 **Features (X)**")
    st.dataframe(X)
    
    y = df['average_score']
    st.write("🎯 **Target (y)**")
    st.dataframe(y)

# ===== Visualisasi =====
with st.expander('📊 Data Visualization'):
    if "gender" in df.columns:
        st.subheader("📊 Rata-rata Skor per Gender")
        avg_by_gender = df.groupby("gender")[["average_score"]].mean()
        st.bar_chart(avg_by_gender)

    if "lunch" in df.columns:
        st.subheader("🍽️ Rata-Rata Skor Berdasarkan Jenis Lunch")
        avg_by_cat = df.groupby("lunch")[["average_score"]].mean().sort_values(by="average_score", ascending=False)
        st.bar_chart(avg_by_cat)

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding Kolom Kategorikal"):
        le = LabelEncoder()

        df['gender'] = le.fit_transform(df['gender'])
        df['lunch'] = le.fit_transform(df['lunch'])
        df['test preparation course'] = le.fit_transform(df['test preparation course'])
        df['race/ethnicity'] = le.fit_transform(df['race/ethnicity'])
        df['parental level of education'] = le.fit_transform(df['parental level of education'])

        st.session_state.df = df
        st.success("✅ Label Encoding selesai.")
        st.dataframe(df)

    if st.button("🔄 Pisahkan Fitur (X) dan Target (y)"):
        X = df.drop(columns=['average_score'])
        y = df['average_score']
        st.write("🧩 X (fitur):")
        st.dataframe(X)
        st.write("🎯 y (target):")
        st.dataframe(y)
