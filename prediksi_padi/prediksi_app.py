import streamlit as st
import pandas as pd
import joblib
import altair as alt
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from src.clustering_prediksi import do_clustering

# === CONFIG TAMPAK LEBAR ===
st.set_page_config(page_title="Prediksi Produksi Padi", layout="wide")

st.title("🌾 Prediksi Produksi Padi – Growth Projection")

# --- INSTRUKSI ATAS ---
st.info(
    "⚠️ Pastikan file Excel memiliki kolom berikut:\n\n"
    "Kecamatan | Komoditas | Tahun | Luas Sawah | Luas Tanam | Luas Panen | Produksi"
)

needed_cols = [
    "Kecamatan", "Komoditas", "Tahun",
    "Luas Sawah", "Luas Tanam", "Luas Panen", "Produksi"
]

# --- INISIALISASI SESSION STATE ---
if "df_proj" not in st.session_state:
    st.session_state.df_proj = None
if "df_clustered" not in st.session_state:
    st.session_state.df_clustered = None
if "prediksi_done" not in st.session_state:
    st.session_state.prediksi_done = False
if "segmentasi_done" not in st.session_state:
    st.session_state.segmentasi_done = False

# === BUAT 2 KOLOM: KIRI (UPLOAD+AKSI) & KANAN (HASIL) ===
col1, col2 = st.columns([1, 2])

#FITUR KIRI
with col1:
    st.header("📂 Upload & Prediksi")

    uploaded_file = st.file_uploader("Unggah data historis (Excel)", type=["xlsx"])

    if uploaded_file is not None:
        df_pred = pd.read_excel(uploaded_file)
        df_pred = df_pred[[c for c in needed_cols if c in df_pred.columns]]
        st.success("✅ Data berhasil dimuat!")
        st.dataframe(df_pred.head())

        model = joblib.load("prediksi_padi/model_rf2.pkl")
        last_year = df_pred["Tahun"].max()
        st.write(f"📅 Data terakhir tersedia: **{last_year}**")

        # === Tombol 1: Jalankan Prediksi ===
        if st.button("🔮 Jalankan Prediksi Produksi"):
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

            # fitur turunan
            df_proj["Rasio_Tanam"] = df_proj["Luas Panen"] / (df_proj["Luas Tanam"] + 1e-9)
            df_proj["Intensitas_Sawah"] = df_proj["Luas Tanam"] / (df_proj["Luas Sawah"] + 1e-9)
            df_proj["Panen_x_Intensitas"] = df_proj["Luas Panen"] * df_proj["Intensitas_Sawah"]
            df_proj["Tanam_x_Rasio"] = df_proj["Luas Tanam"] * df_proj["Rasio_Tanam"]

            # prediksi
            X_proj = df_proj.drop(columns=["Tahun", "Produksi"], errors="ignore")
            df_proj["Prediksi Produksi"] = model.predict(X_proj).round(0)

            # simpan ke session state
            st.session_state.df_proj = df_proj
            st.session_state.prediksi_done = True
            st.success(f"✅ Prediksi Produksi Tahun {last_year + 1} berhasil!")
            st.write("📊 Total Produksi:", int(df_proj["Prediksi Produksi"].sum()))

        # === Tombol 2: Jalankan Segmentasi + Peta ===
        if st.session_state.prediksi_done:
            st.divider()
            st.header("🧩 Segmentasi & Peta")
            if st.button("🗺️ Jalankan Segmentasi + Tampilkan Peta"):
                df_clustered, chart_cluster = do_clustering(st.session_state.df_proj)
                st.session_state.df_clustered = df_clustered
                st.session_state.chart_cluster = chart_cluster
                st.session_state.segmentasi_done = True
                st.success("✅ Segmentasi & Peta siap! Lihat hasil di sebelah kanan ➡️")

#HASIL KANAN
with col2:
    # --- tampilkan hasil prediksi ---
    if st.session_state.prediksi_done and st.session_state.df_proj is not None:
        df_proj = st.session_state.df_proj
        last_year = df_proj["Tahun"].max()

        st.subheader(f"📈 Hasil Prediksi Produksi Tahun {last_year}")
        st.dataframe(df_proj[["Kecamatan", "Tahun", "Prediksi Produksi"]])

        # === VISUALISASI PRODUKSI ===
        st.write("### 📊 Produksi Aktual vs Prediksi")
        df_pred = pd.read_excel(uploaded_file)
        last_year = df_pred["Tahun"].max()
        df_last = df_pred[df_pred["Tahun"] == last_year].set_index("Kecamatan")
        df_actual = df_last.reset_index()[["Kecamatan"]].copy()
        df_actual["Tahun"] = last_year
        df_actual["Produksi"] = df_last["Produksi"].values

        df_prediksi = df_proj[["Kecamatan", "Tahun", "Prediksi Produksi"]].rename(
            columns={"Prediksi Produksi": "Produksi"}
        )
        df_vis = pd.concat([df_actual, df_prediksi])

        chart = (
            alt.Chart(df_vis)
            .mark_bar()
            .encode(
                x=alt.X("Kecamatan:N", sort="-y"),
                y=alt.Y("Produksi:Q"),
                color=alt.Color("Tahun:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Tahun:N",
                tooltip=["Kecamatan", "Tahun", "Produksi"],
            )
            .properties(width=750, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        # total produksi
        st.write("### 📊 Total Produksi: Aktual vs Prediksi")
        df_total = df_vis.groupby("Tahun", as_index=False)["Produksi"].sum()
        chart_total = (
            alt.Chart(df_total)
            .mark_bar()
            .encode(
                x="Tahun:N",
                y="Produksi:Q",
                color="Tahun:N",
                tooltip=["Tahun", "Produksi"],
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(chart_total, use_container_width=True)

    # --- tampilkan hasil segmentasi + peta ---
    if st.session_state.segmentasi_done and st.session_state.df_clustered is not None:
        df_clustered = st.session_state.df_clustered

        st.write("---")
        st.subheader("🧭 Hasil Segmentasi Kecamatan")
        st.dataframe(df_clustered[["Kecamatan", "Cluster", "Prediksi Produksi"]])
        st.altair_chart(st.session_state.chart_cluster, use_container_width=True)

        # === PETA SPASIAL ===
        st.subheader("🗺️ Peta Spasial Cluster Produksi Padi 2025")

        # load geojson
        geo = gpd.read_file("data/sidoarjo_kecamatan.geojson")

        # merge data cluster ke geojson
        merged = geo.merge(df_clustered, on="Kecamatan", how="left")

        # buat peta choropleth
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

        # tooltip hover
        folium.GeoJson(
            merged,
            tooltip=folium.GeoJsonTooltip(
                fields=["Kecamatan", "Cluster", "Prediksi Produksi"],
                aliases=["Kecamatan:", "Cluster:", "Prediksi Produksi (kw):"],
                localize=True,
            ),
        ).add_to(m)

        # tampilkan di Streamlit
        st_folium(m, width=800, height=600)

    if not uploaded_file:
        st.info("⬅️ Unggah file di sebelah kiri untuk mulai prediksi.")
