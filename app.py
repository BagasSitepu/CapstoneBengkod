import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Judul aplikasi
st.title("Visualisasi Data Kualitas Air")

# Membaca data
water_potability_df = pd.read_csv('https://drive.google.com/uc?id=1Nhz5LbYIHC0NTYRAHUvoTf_CF7HFoz_S')

# Menampilkan data frame
st.write("## Data Frame")
st.dataframe(water_potability_df)

# Membuat histogram
st.write("## Histogram")
numerical_features = water_potability_df.select_dtypes(include=['int64', 'float64']).columns
selected_feature = st.selectbox("Pilih fitur numerik:", numerical_features)
fig, ax = plt.subplots()
sns.histplot(water_potability_df[selected_feature], bins=20, color='blue', kde=True, ax=ax)
ax.set_title(f'Distribusi {selected_feature}', fontsize=16)
ax.set_xlabel(selected_feature)
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

# Membuat boxplot
st.write("## Boxplot")
selected_feature = st.selectbox("Pilih fitur numerik untuk boxplot:", numerical_features)
fig, ax = plt.subplots()
sns.boxplot(y=water_potability_df[selected_feature], ax=ax)
ax.set_title(f'Boxplot of {selected_feature}')
ax.set_ylabel(selected_feature)
st.pyplot(fig)

# Membuat heatmap korelasi
st.write("## Heatmap Korelasi")
corr_matrix = water_potability_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title('Korelasi Heatmap untuk Data Kualitas Air')
st.pyplot(fig)

# Data akurasi
models = ['Naive Bayes', 'Decision Tree', 'Random Forest']
initial_accuracies = [53.10, 57.50, 66.90]  # Ganti dengan data akurasi awal Anda
tuned_accuracies = [52.75, 56, 67.12]  # Ganti dengan data akurasi setelah evaluasi

# Membuat visualisasi perbandingan akurasi awal
st.header("Perbandingan Akurasi Awal")
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(models, initial_accuracies, color=['blue', 'green', 'orange'])
ax.set_title('Perbandingan Akurasi Awal')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)  # Sesuaikan dengan rentang akurasi Anda

# Menambahkan label nilai di atas setiap bar
for i, v in enumerate(initial_accuracies):
    ax.text(i, v + 2, str(round(v, 2)) + '%', ha='center', fontweight='bold')

st.pyplot(fig)  # Menampilkan visualisasi di Streamlit

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Non-Potable', 'Potable'],
                yticklabels=['Non-Potable', 'Potable'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    st.pyplot(plt)

# Confusion matrix sebelum normalisasi
st.subheader("Visualisasi Confusion Matrix")

# Data confusion matrix sebelum normalisasi (ganti dengan data Anda)
cm_nb = [[188, 206], [169, 237]]  
cm_dtc = [[221, 173], [167, 239]]
cm_rfc = [[270, 124], [141, 265]]

# Pilihan model
model_choice = st.selectbox("Pilih Model:", ["Naive Bayes", "Decision Tree", "Random Forest"])

# Fungsi untuk menampilkan confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Non-Potable', 'Potable'],
                yticklabels=['Non-Potable', 'Potable'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    st.pyplot(plt)

# Menampilkan confusion matrix berdasarkan pilihan model
if model_choice == "Naive Bayes":
    plot_confusion_matrix(cm_nb, "Naive Bayes")
elif model_choice == "Decision Tree":
    plot_confusion_matrix(cm_dtc, "Decision Tree")
elif model_choice == "Random Forest":
    plot_confusion_matrix(cm_rfc, "Random Forest")

# Membuat visualisasi perbandingan akurasi model
st.header("Perbandingan Akurasi Model (Sebelum & Sesudah Evaluasi)")
x = range(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, initial_accuracies, width, label='Sebelum Evaluasi')
rects2 = ax.bar([i + width for i in x], tuned_accuracies, width, label='Setelah Evaluasi')

ax.set_ylabel('Akurasi (%)')
ax.set_title('Perbandingan Akurasi Model')
ax.set_xticks([i + width / 2 for i in x])
ax.set_xticklabels(models)
ax.legend()

# Menambahkan label nilai di atas setiap bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

st.pyplot(fig)  # Menampilkan visualisasi di Streamlit

st.header("Prediksi Kelayakan Air Minum")

st.write("Pilih Model yang akan digunakan untuk memprediksi kelayakan air minum")
# Muat semua model
models = {
    "Naive Bayes": pickle.load(open("model_naive_bayes.pkl", "rb")),  # Ganti dengan nama file model Naive Bayes
    "Decision Tree": pickle.load(open("model_decision_tree.pkl", "rb")),  # Ganti dengan nama file model Decision Tree
    "Random Forest": pickle.load(open("model_random_forest.pkl", "rb"))  # Ganti dengan nama file model Random Forest
}

# Pilihan model
selected_model = st.selectbox("Pilih Model", list(models.keys()))

st.write("Masukkan nilai untuk masing-masing fitur di bawah ini lalu klik tombol prediksi untuk mengetahui apakah air layak diminum")

# Input fitur
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
Hardness = st.number_input("Hardness", min_value=0.0, max_value=300.0, value=150.0)
Solids = st.number_input("Solids", min_value=0.0, max_value=60000.0, value=20000.0)
Chloramines = st.number_input("Chloramines", min_value=0.0, max_value=15.0, value=7.0)
Sulfate = st.number_input("Sulfate", min_value=0.0, max_value=500.0, value=250.0)
Conductivity = st.number_input("Conductivity", min_value=0.0, max_value=800.0, value=400.0)
Organic_carbon = st.number_input("Organic_carbon", min_value=0.0, max_value=30.0, value=15.0)
Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, max_value=120.0, value=60.0)
Turbidity = st.number_input("Turbidity", min_value=0.0, max_value=7.0, value=3.5)

# Tombol prediksi
if st.button("Prediksi"):
    # Buat DataFrame dari input fitur
    input_data = pd.DataFrame({
        'ph': [ph],
        'Hardness': [Hardness],
        'Solids': [Solids],
        'Chloramines': [Chloramines],
        'Sulfate': [Sulfate],
        'Conductivity': [Conductivity],
        'Organic_carbon': [Organic_carbon],
        'Trihalomethanes': [Trihalomethanes],
        'Turbidity': [Turbidity]
    })

    # Prediksi
    prediction = models[selected_model].predict(input_data)[0]

    # Tampilkan hasil prediksi
    if prediction == 1:
        st.success("Air layak diminum")
    else:
        st.error("Air tidak layak diminum")