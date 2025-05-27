import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.title(':mortar_board: Prediksi Skor Siswa (Math, Reading, Writing)')
st.info('📊 Aplikasi ini memprediksi skor siswa berdasarkan fitur demografis.')

# ===== Inisialisasi =====
if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Visualisasi Awal =====
with st.expander('📁 Visualisasi Awal'):
    st.write("Contoh data:")
    st.dataframe(df)

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding Kolom Kategorikal"):
        le = LabelEncoder()
        for col in ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])
        st.session_state.df = df
        st.success("✅ Label Encoding selesai.")
        st.dataframe(df)

# ===== Training & Evaluation untuk Tiap Skor =====
with st.expander('🧠 Training & Evaluation per Skor'):
    if st.button("🚀 Train Model dan Evaluasi per Skor"):
        df = st.session_state.df
        target_cols = ['math score', 'reading score', 'writing score']
        feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

        # Cek apakah data sudah siap
        if df[feature_cols].select_dtypes(include='object').shape[1] > 0:
            st.error("⛔ Gagal training: Masih ada kolom string! Jalankan Label Encoding terlebih dahulu.")
            st.stop()

        for target in target_cols:
            st.subheader(f"📌 Prediksi: {target.title()}")

            X = df[feature_cols]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write(f"**MAE**: {mae:.2f}")
            st.write(f"**RMSE**: {rmse:.2f}")
            st.write(f"**R² Score**: {r2:.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.set_xlabel("Actual Score")
            ax.set_ylabel("Predicted Score")
            ax.set_title(f"Actual vs Predicted - {target}")
            st.pyplot(fig)
