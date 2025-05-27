import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

st.title(':mortar_board: Student Performance Classification App')
st.info('📊 Klasifikasi apakah siswa lulus atau tidak berdasarkan fitur input.')

# ===== Inisialisasi data hanya sekali (saat awal atau refresh) =====
if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['passed'] = df['average_score'].apply(lambda x: 1 if x >= 60 else 0)
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Data View =====
with st.expander('📁 Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

# ===== Visualisasi =====
with st.expander('📊 Data Visualization'):
    if "lunch" in df.columns:
        st.subheader("🍽️ Persentase Kelulusan Berdasarkan Jenis Lunch")
        avg_by_lunch = df.groupby("lunch")["passed"].mean()
        st.bar_chart(avg_by_lunch)

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

# ===== Training & Evaluation =====
with st.expander('🧠 Training & Evaluation'):
    if st.button("🚀 Train Model dan Evaluasi"):
        df = st.session_state.df
        X = df.drop(columns=['average_score', 'passed'])
        y = df['passed']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.subheader("📈 Evaluasi Model Klasifikasi")
        st.write(f"**Accuracy**: `{acc:.2f}`")
        st.write(f"**Precision**: `{prec:.2f}`")
        st.write(f"**Recall**: `{rec:.2f}`")
        st.write(f"**F1 Score**: `{f1:.2f}`")

        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        st.pyplot(fig)
