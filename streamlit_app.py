import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np

st.title(':mortar_board: Klasifikasi Grade Siswa Berdasarkan Skor Rata-rata')
st.info('📊 Aplikasi ini mengklasifikasikan nilai rata-rata siswa menjadi Grade A–F berdasarkan fitur demografis.')

# ===== Inisialisasi =====
def Grade(avg):
    if avg >= 80: return 'A'
    elif avg >= 70: return 'B'
    elif avg >= 60: return 'C'
    elif avg >= 50: return 'D'
    elif avg >= 40: return 'E'
    else: return 'F'

if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['grade'] = df['average_score'].apply(Grade)
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Visualisasi Awal =====
with st.expander('📁 Visualisasi Awal'):
    st.dataframe(df)

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding + Scaling"):
        le = LabelEncoder()
        df['grade_label'] = le.fit_transform(df['grade'])
        for col in ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        df_encoded = df.copy()
        features = ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']
        scaler = StandardScaler()
        df_encoded[features] = scaler.fit_transform(df_encoded[features])
        st.session_state.df = df_encoded

        st.success("✅ Label Encoding dan StandardScaler selesai diterapkan.")
        st.dataframe(df_encoded)

# ===== Training & Evaluation Grade (Klasifikasi) =====
with st.expander('🧠 Klasifikasi Grade Siswa'):
    df = st.session_state.df
    X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
    y = df['grade_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def run_classification(model, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"📌 {name} - Evaluasi")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

    if st.button("🔘 Train Logistic Regression"):
        run_classification(LogisticRegression(max_iter=1000), "Logistic Regression")

    if st.button("🌳 Train Decision Tree"):
        run_classification(DecisionTreeClassifier(random_state=42), "Decision Tree")

    if st.button("🌲 Train Random Forest"):
        run_classification(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
