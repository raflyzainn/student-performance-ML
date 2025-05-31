import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

# ====== Inisialisasi Aplikasi ======
st.title(':mortar_board: Klasifikasi Grade Siswa Berdasarkan Skor Rata-rata (A–F)')
st.info('📊 Aplikasi ini mengklasifikasikan nilai rata-rata siswa menjadi Grade A–F berdasarkan skor rata-rata.')

if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

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

# ===== Preprocessing =====
with st.expander('🧹 Pre-Processing Data'):
    if st.button("➕ Hitung Rata-rata dan Grade"):
        df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
        df['grade_category'] = df['average_score'].apply(GradeCategory)
        st.session_state.df = df.copy()
        st.success("✅ Kolom average_score dan grade_category berhasil ditambahkan.")
        st.dataframe(df)

    
    if st.button("🧽 Drop Missing Values"):
        df.dropna(inplace=True)
        st.session_state.df = df
        st.success("✅ Missing values dihapus.")
        st.dataframe(df)

    if st.button("🔠 Label Encoding + Scaling"):
        le = LabelEncoder()
        df['grade_label'] = le.fit_transform(df['grade_category'])

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

# ===== Model Manual =====
def custom_logistic_regression(X_train, y_train, X_test):
    classes = np.unique(y_train)
    X_train = np.insert(X_train.values, 0, 1, axis=1)
    X_test = np.insert(X_test.values, 0, 1, axis=1)
    y_train = y_train.values

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train_binary_logreg(X, y, lr=0.01, epochs=300):
        weights = np.zeros((X.shape[1],))
        for _ in range(epochs):
            z = np.dot(X, weights)
            pred = sigmoid(z)
            gradient = np.dot(X.T, (pred - y)) / len(y)
            weights -= lr * gradient
        return weights

    weight_dict = {}
    for cls in classes:
        y_binary = np.where(y_train == cls, 1, 0)
        weight_dict[cls] = train_binary_logreg(X_train, y_binary)

    preds = []
    for x in X_test:
        probs = {cls: sigmoid(np.dot(x, weight_dict[cls])) for cls in classes}
        pred_class = max(probs, key=probs.get)
        preds.append(pred_class)
    return np.array(preds)

def custom_knn(X_train, y_train, X_test, k=5):
    from scipy.spatial.distance import cdist
    distances = cdist(X_test, X_train, 'euclidean')
    neighbors = np.argsort(distances, axis=1)[:, :k]
    preds = []
    for n in neighbors:
        votes = y_train.iloc[n]
        preds.append(np.bincount(votes).argmax())
    return np.array(preds)

def custom_naive_bayes(X_train, y_train, X_test):
    classes = np.unique(y_train)
    summaries = {}
    for cls in classes:
        features = X_train[y_train == cls]
        summaries[cls] = [(np.mean(col), np.std(col)) for col in features.T]

    def calculate_probability(x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def predict_single(x):
        probabilities = {}
        for cls, stats in summaries.items():
            prob = 1
            for i in range(len(stats)):
                mean, stdev = stats[i]
                prob *= calculate_probability(x[i], mean, stdev)
            probabilities[cls] = prob
        return max(probabilities, key=probabilities.get)

    return np.array([predict_single(x) for x in X_test])

# ===== Training & Evaluation Grade (Klasifikasi) =====
with st.expander('🧠 Klasifikasi Grade Siswa'):
    df_final = st.session_state.df
    if 'grade_label' not in df_final.columns:
        st.error("⛔ Kolom 'grade_label' belum ada. Harap jalankan Label Encoding terlebih dahulu.")
        st.stop()

    X = df_final[['math score', 'reading score', 'writing score',
                  'gender', 'lunch', 'test preparation course', 'race/ethnicity', 'parental level of education']]
    y = df_final['grade_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(y_test, y_pred, name):
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        st.subheader(f"📌 {name} - Evaluasi")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.write(f"**Macro Precision:** {prec:.2f}")
        st.write(f"**Macro Recall:** {rec:.2f}")
        st.write(f"**Macro F1-Score:** {f1:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)

        st.session_state.model_results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        }

    if st.button("🔘 Train Custom Logistic Regression"):
        y_pred = custom_logistic_regression(X_train, y_train, X_test)
        evaluate_model(y_test, y_pred, "Custom Logistic Regression")

    if st.button("📌 Train K-NN (Manual)"):
        y_pred = custom_knn(X_train.to_numpy(), y_train, X_test.to_numpy(), k=5)
        evaluate_model(y_test, y_pred, "Custom K-NN")

    if st.button("🧠 Train Naive Bayes"):
        y_pred = custom_naive_bayes(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy())
        evaluate_model(y_test, y_pred, "Custom Naive Bayes")

# ===== Tab Evaluasi Model =====
with st.expander("📊 Evaluasi Model - Komparasi"):
    if st.session_state.model_results:
        results_df = pd.DataFrame(st.session_state.model_results).T
        st.subheader("📈 Tabel Perbandingan Model")
        st.dataframe(results_df.style.highlight_max(axis=0))

        st.subheader("📉 Grafik Perbandingan Metrik Model")
        metric = st.selectbox("Pilih metrik untuk visualisasi", ['Accuracy', 'Precision', 'Recall', 'F1-Score'])

        fig, ax = plt.subplots()
        results_df[metric].plot(kind='bar', color='skyblue', ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f'Perbandingan {metric} antar Model')
        st.pyplot(fig)
    else:
        st.warning("❗ Belum ada model yang dilatih. Silakan jalankan minimal 1 model dulu.")
