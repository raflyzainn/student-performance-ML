import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import seaborn as sns
from io import StringIO

st.title(':mortar_board: Klasifikasi Grade Siswa Berdasarkan Skor Rata-rata (A–F)')
st.info('📊 Aplikasi ini mengklasifikasikan nilai rata-rata siswa menjadi Grade A–F berdasarkan skor rata-rata.')

# ===== Inisialisasi Grade =====
def GradeCategory(avg):
    if avg >= 90: return 'A'
    elif avg >= 80: return 'B'
    elif avg >= 70: return 'C'
    elif avg >= 60: return 'D'
    elif avg >= 50: return 'E'
    else: return 'F'

if 'df' not in st.session_state:
    df = pd.read_csv("StudentsPerformance.csv")
    df.columns = [str(col).strip() for col in df.columns]
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
    df['grade_category'] = df['average_score'].apply(GradeCategory)
    st.session_state.df = df.copy()

df = st.session_state.df

# ===== Visualisasi Awal =====
with st.expander('📁 Data'):
    st.subheader("🗃️ Tabel Data Siswa")
    st.dataframe(df)

    st.subheader("📑 Informasi Umum Dataset")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.subheader("📊 Statistik Deskriptif")
    st.dataframe(df.describe())

    st.subheader("🔢 Jumlah Data Tiap Grade")
    st.dataframe(df['grade_category'].value_counts())


# ===== Visualisasi Data =====
with st.expander('📊 Data Visualization'):
    st.subheader("📈 Distribusi Nilai Rata-rata Siswa")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['average_score'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel('Average Score')
    ax.set_ylabel('Jumlah Siswa')
    ax.set_title('Histogram Nilai Rata-rata')
    st.pyplot(fig)

    st.subheader("📌 Korelasi Antar Fitur")
    numeric_features = ['math score', 'reading score', 'writing score', 'average_score']
    corr = df[numeric_features].corr()
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Heatmap Korelasi')
    st.pyplot(fig_corr)

    st.subheader("🎯 Distribusi Grade Siswa")
    fig_grade, ax_grade = plt.subplots(figsize=(8, 4))
    sns.countplot(x='grade_category', data=df, order=['A', 'B', 'C', 'D', 'E', 'F'], palette='viridis', ax=ax_grade)
    ax_grade.set_xlabel('Grade')
    ax_grade.set_ylabel('Jumlah Siswa')
    ax_grade.set_title('Distribusi Grade Siswa')
    st.pyplot(fig_grade)
    
# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding + Scaling"):
        le = LabelEncoder()
        df['grade_label'] = le.fit_transform(df['grade_category'])  # A=0, B=1, C=2, dst.

        for col in ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])

        features = ['gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education',
                    'math score', 'reading score', 'writing score']

        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

        st.session_state.df = df.copy()
        st.success("✅ Label Encoding dan StandardScaler selesai diterapkan.")
        st.dataframe(df)

# ===== Training & Evaluation Grade (Klasifikasi) =====
with st.expander('🧠 Klasifikasi Grade Siswa'):
    df_final = st.session_state.df_smote if 'df_smote' in st.session_state else st.session_state.df

    X = df_final[['math score', 'reading score', 'writing score',
                  'gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']]

    if 'grade_label' not in df_final.columns:
        st.error("⛔ Kolom 'grade_label' belum ada. Harap jalankan Label Encoding terlebih dahulu.")
        st.stop()

    y = df_final['grade_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def run_classification(model, name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader(f"📌 {name} - Evaluasi")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

    if st.button("🔘 Train Logistic Regression"):
        run_classification(LogisticRegression(max_iter=1000), "Logistic Regression")

    if st.button("🌳 Train Decision Tree"):
        run_classification(DecisionTreeClassifier(random_state=42), "Decision Tree")

    if st.button("🌲 Train Random Forest"):
        run_classification(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
