# src/clustering_app.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import altair as alt

def do_clustering(df, n_clusters=3):
    """
    Segmentasi kecamatan berdasarkan hasil prediksi produksi
    dan fitur-fitur pertanian terkait.
    """

    # --- pastikan semua kolom ada ---
    needed = [
        "Kecamatan","Luas Sawah","Luas Tanam","Luas Panen",
        "Rasio_Tanam","Intensitas_Sawah",
        "Panen_x_Intensitas","Tanam_x_Rasio",
        "Prediksi Produksi"
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di DataFrame")

    # --- ambil fitur numerik ---
    feature_cols = needed[1:]  # skip Kecamatan
    X = df[feature_cols].values

    # --- normalisasi data biar KMeans tidak bias ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- kmeans clustering ---
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # --- simpan hasil ke dataframe (jangan di-overwrite lagi) ---
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    # mapping sesuai interpretasi kamu
    # 0 = unggul, 1 = kecil, 2 = sedang
    cluster_name_map = {
        0: "Unggul",
        1: "Kecil",
        2: "Sedang"
    }

    # label yang lebih deskriptif (buat BI/dashboard)
    cluster_desc_map = {
        0: "Cluster Unggul",
        1: "Cluster Kecil",
        2: "Cluster Sedang"
    }

    df_clustered["ClusterName"] = df_clustered["Cluster"].map(cluster_name_map)
    df_clustered["Kategori"] = df_clustered["Cluster"].map(cluster_desc_map)

    # --- reduksi dimensi dengan PCA (untuk visual) ---
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_clustered["PC1"] = X_pca[:, 0]
    df_clustered["PC2"] = X_pca[:, 1]

    # --- visualisasi (pakai ClusterName biar kebaca, bukan 0/1/2) ---
    chart = (
        alt.Chart(df_clustered)
        .mark_circle(size=200)
        .encode(
            x=alt.X("PC1:Q", title="Komponen Utama 1"),
            y=alt.Y("PC2:Q", title="Komponen Utama 2"),
            color=alt.Color("ClusterName:N", scale=alt.Scale(scheme="tableau10"), title="Klaster"),
            tooltip=[
                "Kecamatan",
                "ClusterName",
                "Prediksi Produksi",
                "Luas Panen",
                "Luas Tanam",
                "Luas Sawah",
                "Rasio_Tanam",
                "Intensitas_Sawah"
            ]
        )
        .properties(width=700, height=400, title="📊 Segmentasi Kecamatan (PCA 2D Projection)")
    )

    return df_clustered, chart
