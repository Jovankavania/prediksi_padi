import streamlit as st
import pandas as pd
import joblib
import altair as alt
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from src.clustering_prediksi import do_clustering

# === CONFIG ===
st.set_page_config(page_title="Prediksi Produksi Padi", page_icon="🌾", layout="wide")

# --- Sidebar (Panel Kiri) ---
st.sidebar.title("📂 Upload & Prediksi Data")
st.sidebar.info("Pastikan file Excel memiliki kolom berikut:\n\n"
                "Kecamatan | Komoditas | Tahun | Luas Sawah | Luas Tanam | Luas Panen | Produksi")

uploaded_file = st.sidebar.file_uploader("Unggah file Excel", type=["xlsx"])
prediksi_button = False
segmentasi_button = False

if uploaded_file is not None:
    st.sidebar.success("✅ File berhasil diunggah.")
    prediksi_button = st.sidebar.button("🔮 Jalankan Prediksi")
    segmentasi_button = st.sidebar.button("🧩 Segmentasi + Peta")

# --- Area Utama (Kanan) ---
st.title("🌾 Prediksi Produksi Padi – Growth Projection")

if uploaded_file is None:
    st.info("⬅️ Unggah file Excel di panel kiri untuk mulai prediksi.")
else:
    df_pred = pd.read_excel(uploaded_file)
    model = joblib.load("prediksi_padi/model_rf2.pkl")

    last_year = df_pred["Tahun"].max()

    if prediksi_button:
        growth = (
            df_pred[df_pred["Tahun"].between(last_year - 2, last_year)]
            .groupby("Kecamatan")[["Luas Sawah", "Luas Tanam", "Luas Panen"]]
            .pct_change()
            .clip(-0.1, 0.1)
            .groupby(df_pred["Kecamatan"])
            .mean()
        )
        df_last = df_pred[df_pred["Tahun"] == last_year].set_index("Kecamatan")
        df_proj = df_last.copy()
        df_proj[["Luas Sawah", "Luas Tanam", "Luas Panen"]] *= (1 + growth)
        df_proj["Tahun"] = last_year + 1
        df_proj = df_proj.reset_index()

        df_proj["Rasio_Tanam"] = df_proj["Luas Panen"] / (df_proj["Luas Tanam"] + 1e-9)
        df_proj["Intensitas_Sawah"] = df_proj["Luas Tanam"] / (df_proj["Luas Sawah"] + 1e-9)
        df_proj["Panen_x_Intensitas"] = df_proj["Luas Panen"] * df_proj["Intensitas_Sawah"]
        df_proj["Tanam_x_Rasio"] = df_proj["Luas Tanam"] * df_proj["Rasio_Tanam"]

        X_proj = df_proj.drop(columns=["Tahun", "Produksi"], errors="ignore")
        df_proj["Prediksi Produksi"] = model.predict(X_proj).round(0)

        st.session_state.df_proj = df_proj
        st.success(f"✅ Prediksi Produksi Tahun {last_year + 1} selesai!")
        st.dataframe(df_proj[["Kecamatan", "Tahun", "Prediksi Produksi"]])
        st.write("📊 Total Produksi:", int(df_proj["Prediksi Produksi"].sum()))

        # Visualisasi bar chart
        df_prediksi = df_proj[["Kecamatan", "Prediksi Produksi"]]
        chart = (
            alt.Chart(df_prediksi)
            .mark_bar(color="#007bff")
            .encode(
                x=alt.X("Kecamatan:N", sort="-y"),
                y="Prediksi Produksi:Q",
                tooltip=["Kecamatan", "Prediksi Produksi"]
            )
            .properties(width=800, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

    if segmentasi_button and "df_proj" in st.session_state:
        df_clustered, chart_cluster = do_clustering(st.session_state.df_proj)
        st.session_state.df_clustered = df_clustered
        st.session_state.chart_cluster = chart_cluster

        st.subheader("🧭 Hasil Segmentasi Kecamatan")
        st.dataframe(df_clustered[["Kecamatan", "Cluster", "Prediksi Produksi"]])
        st.altair_chart(chart_cluster, use_container_width=True)

        # === PETA ===
        st.subheader("🗺️ Peta Spasial Cluster Produksi Padi")
        geo = gpd.read_file("data/sidoarjo_kecamatan.geojson")
        merged = geo.merge(df_clustered, on="Kecamatan", how="left")

        m = folium.Map(location=[-7.45, 112.7], zoom_start=11)
        folium.Choropleth(
            geo_data=merged,
            data=merged,
            columns=["Kecamatan", "Cluster"],
            key_on="feature.properties.Kecamatan",
            fill_color="Set1",
            fill_opacity=0.8,
            line_opacity=0.3,
            legend_name="Cluster Potensi Produksi",
        ).add_to(m)

        folium.GeoJson(
            merged,
            tooltip=folium.GeoJsonTooltip(
                fields=["Kecamatan", "Cluster", "Prediksi Produksi"],
                aliases=["Kecamatan:", "Cluster:", "Prediksi Produksi (kw):"],
                localize=True,
            ),
        ).add_to(m)

        st_folium(m, width=900, height=600)
