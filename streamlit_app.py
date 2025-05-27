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
from imblearn.over_sampling import SMOTE

st.title(':mortar_board: Klasifikasi Grade Siswa Berdasarkan Skor Rata-rata')
st.info('📊 Aplikasi ini mengklasifikasikan nilai rata-rata siswa menjadi Grade A–F berdasarkan fitur demografis.')

# ===== Inisialisasi =====
def GradeCategory(avg):
    if avg >= 70: return 'High'     # A, B
    elif avg >= 50: return 'Medium' # C, D
    else: return 'Low'              # E, F


if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['grade_category'] = df['average_score'].apply(GradeCategory)
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Visualisasi Awal =====
with st.expander('📁 Visualisasi Awal'):
    st.dataframe(df)

from imblearn.over_sampling import SMOTE

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding + Scaling"):
        le = LabelEncoder()
        df['grade_category'] = df['average_score'].apply(GradeCategory)
        df['grade_label'] = le.fit_transform(df['grade_category'])  # 0 = Low, 1 = Medium, 2 = High

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

    if st.button("⚖️ SMOTE - Atasi Imbalance"):
        df_encoded = st.session_state.df
        if 'grade_label' not in df_encoded.columns:
            st.error("⛔ Kolom 'grade_label' belum ada. Jalankan Label Encoding terlebih dahulu.")
            st.stop()

        features = ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']
        X = df_encoded[features]
        y = df_encoded['grade_label']

        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)

        df_smote = pd.DataFrame(X_smote, columns=features)
        df_smote['grade_label'] = y_smote

        st.session_state.df_smote = df_smote

        st.success("✅ SMOTE selesai diterapkan. Data imbalance sudah diatasi.")
        st.dataframe(df_smote)

# ===== Training & Evaluation Grade (Klasifikasi) =====
with st.expander('🧠 Klasifikasi Grade Siswa'):
    df = st.session_state.df
    X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
    
    if 'grade_label' not in df.columns:
        st.error("⛔ Kolom 'grade_label' belum ada. Harap jalankan Label Encoding + Scaling terlebih dahulu.")
        st.stop()

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
