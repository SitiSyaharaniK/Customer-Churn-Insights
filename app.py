import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.graph_objects as go

# 1. Set konfigurasi halaman web (Theme modern & responsif)
st.set_page_config(
    page_title="Customer Churn Insights", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ambil path direktori tempat app.py berada agar aman dari FileNotFoundError
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Memuat model dan scaler 
@st.cache_resource
def load_models():
    # Menangani variasi nama file jika ada spasi bawaan
    possible_model_names = ['best_churn_model.pkl', 'best_churn_model .pkl']
    possible_scaler_names = ['dataset_scaler.pkl', 'dataset_scaler .pkl']
    
    model_path = None
    scaler_path = None
    
    for name in possible_model_names:
        if os.path.exists(os.path.join(BASE_DIR, name)):
            model_path = os.path.join(BASE_DIR, name)
            break
            
    for name in possible_scaler_names:
        if os.path.exists(os.path.join(BASE_DIR, name)):
            scaler_path = os.path.join(BASE_DIR, name)
            break

    if not model_path or not scaler_path:
        # Fallback default jika tidak terdeteksi otomatis
        model_path = os.path.join(BASE_DIR, 'best_churn_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'dataset_scaler.pkl')
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    best_churn_model, dataset_scaler = load_models()
    model_features = dataset_scaler.feature_names_in_
except Exception as e:
    st.error(f"Gagal memuat model/scaler. Pastikan file berada di folder yang sama dengan app.py. Error: {e}")
    st.stop()

# --- SIDEBAR: FORMULIR INPUT DATA PELANGGAN ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80) 
    st.title("Data Pelanggan")
    st.write("Silakan sesuaikan metrik pelanggan di bawah ini:")
    st.markdown("---")
    
    # Kelompok 1: Profil & Hubungan
    st.subheader(" Profil & Hubungan")
    tenure = st.number_input("Masa Berlangganan (Bulan)", min_value=0, max_value=120, value=12)
    age = st.number_input("Umur Pelanggan", min_value=10, max_value=100, value=30)
    satisfaction_score = st.slider("Skor Kepuasan (1-5)", min_value=1, max_value=5, value=4)
    
    st.markdown("---")
    # Kelompok 2: Aktivitas Platform
    st.subheader(" Aktivitas Platform")
    total_visits = st.number_input("Total Kunjungan", min_value=0, value=10)
    avg_session_time = st.number_input("Rata-rata Durasi Sesi (Menit)", min_value=0.0, value=15.0)
    
    st.markdown("---")
    # Kelompok 3: Finansial & Nilai Kontrak
    st.subheader(" Informasi Finansial")
    monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Tagihan Sistem ($)", min_value=0.0, value=600.0)
    total_spent = st.number_input("Total Pengeluaran ($)", min_value=0.0, value=600.0)
    lifetime_value = st.number_input("Estimasi CLV ($)", min_value=0.0, value=1000.0)
    
    st.markdown("---")
    predict_btn = st.button("Hitung Analisis Prediksi", use_container_width=True, type="primary")


# --- HALAMAN UTAMA: DASHBOARD UTAMA ---
st.title(" Customer Churn Analytics Dashboard")
st.markdown("Aplikasi prediksi berbasis kecerdasan buatan untuk mengidentifikasi risiko loyalitas pelanggan secara *real-time*.")
st.markdown("---")

if predict_btn:
    # A. Proses Data Input
    user_data = {
        'tenure': tenure,
        'age': age,
        'total_visits': total_visits,
        'MonthlyCharges': monthly_charges,
        'total_spent': total_spent,
        'avg_session_time': avg_session_time,
        'TotalCharges': total_charges,
        'lifetime_value': lifetime_value,
        'satisfaction_score': satisfaction_score
    }
    
    input_user_df = pd.DataFrame([user_data])
    input_final_df = input_user_df.reindex(columns=model_features, fill_value=0)
    input_scaled = dataset_scaler.transform(input_final_df)
    
    # B. Eksekusi Prediksi
    prediction = best_churn_model.predict(input_scaled)
    probability = best_churn_model.predict_proba(input_scaled)[0][1]
    prob_percentage = probability * 100
    
    # C. TAMPILAN VISUAL UTAMA
    st.subheader(" Hasil Analisis Inteligensi Model")
    
    # Membagi layout: Kolom Kiri untuk Ringkasan & Gauge, Kolom Kanan untuk Status & Rekomendasi
    layout_col1, layout_col2 = st.columns([2, 3])
    
    with layout_col1:
        # Membuat visualisasi Gauge Chart menggunakan Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Tingkat Risiko Churn (%)", 'font': {'size': 18}},
            number = {'suffix': "%", 'font': {'size': 36}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "#333333"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 35], 'color': '#2ecc71'},   # Hijau: Aman
                    {'range': [35, 65], 'color': '#f1c40f'},  # Kuning: Waspada
                    {'range': [65, 100], 'color': '#e74c3c'}  # Merah: Bahaya
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_percentage
                }
            }
        ))
        
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with layout_col2:
        # Menampilkan status ringkas berbasis KPI Card
        kpi_1, kpi_2 = st.columns(2)
        
        if prediction[0] == 1:
            kpi_1.metric(label="Status Akhir", value="🔴 CHURN")
            kpi_2.metric(label="Evaluasi Kepuasan", value=f"{satisfaction_score} / 5")
            
            st.error(f" **Peringatan Tinggi:** Pelanggan ini diprediksi berisiko kuat untuk **Keluar / Berhenti Berlangganan**.")
            
            with st.expander(" Rekomendasi Tindakan Mitigasi", expanded=True):
                st.write(f"""
                - **Intervensi Khusus:** Segera lakukan *follow-up* personal. Pertimbangkan pemberian insentif karena tagihan bulanannya mencapai **${monthly_charges:.2f}**.
                - **Optimalisasi Aktivitas:** Skor kepuasan pelanggan berada di angka **{satisfaction_score}**. Periksa apakah durasi sesi (**{avg_session_time} mnt**) mengalami hambatan teknis.
                """)
        else:
            kpi_1.metric(label="Status Akhir", value="🟢 LOYAL")
            kpi_2.metric(label="Evaluasi Kepuasan", value=f"{satisfaction_score} / 5")
            
            st.success(f"**Kabar Baik:** Pelanggan ini diprediksi akan **Tetap Setia dan Melanjutkan Langganan**.")
            
            with st.expander("📌 Strategi Retensi & Upselling", expanded=True):
                st.write(f"""
                - **Apresiasi Pelanggan:** Nilai kontribusi masa depan pelanggan (CLV) diproyeksikan sebesar **${lifetime_value:.2f}**. Masukkan ke dalam program loyalitas eksklusif.
                - **Strategi Monetisasi:** Karena tingkat risiko sangat rendah (**{prob_percentage:.2f}%**), ini momen ideal untuk menawarkan peningkatan paket (*upgrade tier*).
                """)

else:
    # Tampilan Panduan Awal (Placeholder)
    st.info(" Silakan isi parameter data pelanggan di menu sidebar sebelah kiri, kemudian klik tombol **'Hitung Analisis Prediksi'**.")
    
    st.subheader("Indikator Utama Nilai Prediksi")
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.help("**Masa Berlangganan (Tenure)**: Menghitung total durasi ikatan kontrak pelanggan dalam satuan bulan.")
    p_col2.help("**Skor Kepuasan (Satisfaction)**: Representasi penilaian subjektif yang diberikan pelanggan terhadap kualitas platform.")
    p_col3.help("**Estimasi CLV (Customer Lifetime Value)**: Total nilai finansial jangka panjang yang berpotensi diberikan pelanggan kepada bisnis.")