import streamlit as st
import pandas as pd
import joblib
import altair as alt

# === CONFIG TAMPAK LEBAR ===
st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")

# === SEDIKIT CSS CUSTOM ===
st.markdown("""
    <style>
        /* Hilangkan margin default dan lebar penuh */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        /* Panel kiri */
        .left-panel {
            background-color: #f9fafb;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
            height: 100%;
        }
        /* Heading kanan */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        /* Subheader biru */
        .info-box {
            background-color: #e8f0fe;
            padding: 0.8rem 1rem;
            border-radius: 8px;
            color: #1a4fb7;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

# === HEADER APLIKASI ===
st.markdown("""
<div class="main-header">🌾 Prediksi Produksi Padi – Growth Projection</div>
<div class="info-box">
⚠️ Pastikan file Excel memiliki kolom berikut:<br>
<b>Kecamatan | Komoditas | Tahun | Luas Sawah | Luas Tanam | Luas Panen | Produksi</b>
</div>
""", unsafe_allow_html=True)

# === BUAT 2 KOLOM: KIRI (UPLOAD) & KANAN (HASIL) ===
col1, col2 = st.columns([1, 2], gap="large")

# =========================
# KOLOM KIRI: PANEL UPLOAD
# =========================
with col1:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.header("📂 Upload & Prediksi")

    uploaded_file = st.file_uploader("Unggah data historis (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.success("✅ File berhasil diunggah!")
        st.dataframe(df.head())
    else:
        st.info("⬅️ Unggah file di sini untuk mulai prediksi.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# KOLOM KANAN: KONTEN UTAMA
# =========================
with col2:
    st.markdown("### 📈 Hasil Prediksi & Visualisasi")
    st.write("Hasil prediksi akan muncul di sini nanti setelah data diunggah.")
    st.empty()
