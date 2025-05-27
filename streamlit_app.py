import streamlit as st
import pandas as pd

st.title(':mortar_board: Student Performance in Exams by Machine Learning')
st.info('📊 Student Performance Analysis by Machine Learning')

# Load & Siapkan Data
df = pd.read_csv("StudentsPerformance.csv")
df.columns = [str(col).strip() for col in df.columns]
df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

# 🔘 Navigasi tahap (radio button atau selectbox bisa juga)
step = st.radio("Pilih Tahap Analisis:", [
    "📥 Lihat Data", 
    "📊 Visualisasi Data", 
    "🧹 Preprocessing"
])

# 📥 TAHAP 1: Lihat Data
if step == "📥 Lihat Data":
    with st.expander('📁 Data'):
        st.write('**Raw Data**')
        st.dataframe(df)

        # Fitur
        X = df.drop(columns=['average_score'])
        st.write("🧩 **Features (X)**")
        st.dataframe(X)

        # Target
        y = df['average_score']
        st.write("🎯 **Target (y)**")
        st.dataframe(y)

# 📊 TAHAP 2: Visualisasi
elif step == "📊 Visualisasi Data":
    with st.expander('📈 Visualisasi'):
        avg_by_gender = df.groupby("gender")[["average_score"]].mean()
        st.subheader("📊 Rata-rata Skor per Gender")
        st.bar_chart(avg_by_gender)

# 🧹 TAHAP 3: Preprocessing
elif step == "🧹 Preprocessing":
    st.subheader("🧹 Data Preprocessing")
    
    # Encoding kategorikal
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    st.write("✅ Kolom setelah Encoding:")
    st.write(df_encoded.columns.tolist())

    # Pisahkan X dan y
    X = df_encoded.drop(columns=['average_score'])
    y = df_encoded['average_score']
    
    st.write("🧩 X (fitur):", X.shape)
    st.write("🎯 y (target):", y.shape)

    st.dataframe(X.head())
    st.dataframe(y.head())
