import streamlit as st
import pandas as pd
import joblib
import altair as alt
from src.clustering_prediksi import do_clustering   # ✅ pastikan path-nya sesuai strukturmu

st.title("🌾 Prediksi Produksi Padi – Growth Projection")

# 👉 Pengingat struktur tabel
st.info("⚠️ Pastikan file Excel punya kolom berikut:\n\n"
        "Kecamatan | Komoditas | Tahun | Luas Sawah | Luas Tanam | Luas Panen | Produksi")

# daftar kolom wajib
needed_cols = ["Kecamatan","Komoditas","Tahun","Luas Sawah","Luas Tanam","Luas Panen","Produksi"]

# 1. Upload file Excel
uploaded_file = st.file_uploader("Unggah data historis (Excel)", type=["xlsx"])

if uploaded_file is not None:
    df_pred = pd.read_excel(uploaded_file)

    # filter hanya kolom yang dibutuhkan
    df_pred = df_pred[[c for c in needed_cols if c in df_pred.columns]]

    st.write("✅ Data berhasil dimuat (kolom lain otomatis dihapus):")
    st.dataframe(df_pred.head())

    # 2. Load model
    model = joblib.load("prediksi_padi/model_rf2.pkl")

    # 3. Tahun terakhir
    last_year = df_pred["Tahun"].max()
    st.write(f"Data terakhir tersedia: **{last_year}**")

    if st.button("🔮 Proyeksikan Tahun Berikutnya"):
        # hitung growth rata-rata
        growth = (
                df_pred[df_pred["Tahun"].between(last_year-2, last_year)]
                .groupby("Kecamatan")[["Luas Sawah","Luas Tanam","Luas Panen"]].pct_change()
                .clip(-0.1, 0.1)   #batasi 10%
                .groupby(df_pred["Kecamatan"]).mean()
        )

        df_last = df_pred[df_pred["Tahun"] == last_year].set_index("Kecamatan")
        df_proj = df_last.copy()
        df_proj[["Luas Sawah","Luas Tanam","Luas Panen"]] *= (1 + growth)

        df_proj["Tahun"] = last_year + 1
        df_proj = df_proj.reset_index()

        # fitur turunan
        df_proj["Rasio_Tanam"] = df_proj["Luas Panen"] / (df_proj["Luas Tanam"]+1e-9)
        df_proj["Intensitas_Sawah"] = df_proj["Luas Tanam"] / (df_proj["Luas Sawah"]+1e-9)
        df_proj["Panen_x_Intensitas"] = df_proj["Luas Panen"] * df_proj["Intensitas_Sawah"]
        df_proj["Tanam_x_Rasio"]      = df_proj["Luas Tanam"] * df_proj["Rasio_Tanam"]

        # prediksi
        X_proj = df_proj.drop(columns=["Tahun","Produksi"], errors="ignore")
        y_pred_proj = model.predict(X_proj)
        df_proj["Prediksi Produksi"] = y_pred_proj.round(0)

        st.success(f"✅ Prediksi Produksi Padi Tahun {last_year+1}")
        st.dataframe(df_proj[["Kecamatan","Tahun","Prediksi Produksi"]])
        st.write("📊 Total Produksi:", int(df_proj["Prediksi Produksi"].sum()))

        # === VISUALISASI (Altair side-by-side) ===
        df_actual = df_last.reset_index()[["Kecamatan"]].copy()
        df_actual["Tahun"] = last_year
        df_actual["Produksi"] = df_last["Produksi"].values

        df_prediksi = df_proj[["Kecamatan","Tahun","Prediksi Produksi"]].rename(
            columns={"Prediksi Produksi": "Produksi"}
        )

        df_vis = pd.concat([df_actual, df_prediksi])

        st.write("### 📈 Perbandingan Produksi Aktual vs Prediksi")
        chart = (
            alt.Chart(df_vis)
            .mark_bar()
            .encode(
                x=alt.X("Kecamatan:N", sort="-y"),
                y=alt.Y("Produksi:Q"),
                color=alt.Color("Tahun:N", scale=alt.Scale(scheme="tableau10")),
                xOffset="Tahun:N",  # 👉 side-by-side
                tooltip=["Kecamatan", "Tahun", "Produksi"]
            )
            .properties(width=700, height=400)
        )
        st.altair_chart(chart, use_container_width=True)

        # total produksi bar chart
        st.write("### 📊 Total Produksi: Aktual vs Prediksi")
        df_total = df_vis.groupby("Tahun", as_index=False)["Produksi"].sum()
        chart_total = (
            alt.Chart(df_total)
            .mark_bar()
            .encode(
                x="Tahun:N",
                y="Produksi:Q",
                color="Tahun:N",
                tooltip=["Tahun", "Produksi"]
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(chart_total, use_container_width=True)

        # === 🔹 Tambahkan Tombol Segmentasi Setelah Prediksi ===
        st.write("---")
        st.subheader("🔍 Segmentasi Hasil Prediksi Produksi")
        if st.button("🧩 Lakukan Segmentasi Hasil Prediksi"):
            df_clustered, chart_cluster = do_clustering(df_proj)
            st.write("### 📊 Hasil Segmentasi Kecamatan")
            st.dataframe(df_clustered[["Kecamatan", "Cluster", "Prediksi Produksi"]])
            st.altair_chart(chart_cluster, use_container_width=True)

else:
    st.info("⬆️ Silakan unggah file Excel untuk mulai prediksi.")
