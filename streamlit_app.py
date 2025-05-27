import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.title(':mortar_board: Prediksi Skor Siswa (Math, Reading, Writing)')
st.info('📊 Memprediksi skor ujian siswa berdasarkan fitur demografis.')

# ===== Inisialisasi data hanya sekali =====
if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Visualisasi =====
with st.expander('📊 Visualisasi Awal'):
    st.dataframe(df.head())

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing'):
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

# ===== Training & Evaluation per Target =====
with st.expander('🧠 Training & Evaluation per Skor'):
    if st.button("🚀 Train dan Evaluasi Model untuk Masing-masing Skor"):
        df = st.session_state.df

        feature_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        target_cols = ['math score', 'reading score', 'writing score']

        for target in target_cols:
            st.subheader(f"📈 Prediksi: {target.title()}")

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
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{target.title()} - Actual vs Predicted")
            st.pyplot(fig)
