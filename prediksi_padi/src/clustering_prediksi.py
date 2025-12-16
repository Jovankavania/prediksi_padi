# src/clustering_prediksi.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import altair as alt

def do_clustering(df, n_clusters=3):
    needed = [
        "Kecamatan","Luas Sawah","Luas Tanam","Luas Panen",
        "Rasio_Tanam","Intensitas_Sawah",
        "Panen_x_Intensitas","Tanam_x_Rasio",
        "Prediksi Produksi"
    ]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ditemukan di DataFrame")

    feature_cols = needed[1:]  # skip Kecamatan
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    # ✅ Label klaster sesuai interpretasi kamu
    label_map = {0: "Unggul", 1: "Kecil", 2: "Sedang"}
    df_clustered["Cluster_Label"] = df_clustered["Cluster"].map(label_map)

    # PCA untuk visualisasi
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_clustered["PC1"] = X_pca[:, 0]
    df_clustered["PC2"] = X_pca[:, 1]

    # Visualisasi: warna pakai label biar kebaca
    chart = (
        alt.Chart(df_clustered)
        .mark_circle(size=200)
        .encode(
            x=alt.X("PC1:Q", title="Komponen Utama 1"),
            y=alt.Y("PC2:Q", title="Komponen Utama 2"),
            color=alt.Color("Cluster_Label:N", title="Klaster"),
            tooltip=[
                "Kecamatan",
                "Cluster",
                "Cluster_Label",
                "Prediksi Produksi",
                "Luas Panen",
                "Luas Tanam",
                "Luas Sawah",
                "Rasio_Tanam",
                "Intensitas_Sawah"
            ],
        )
        .properties(width=700, height=400, title="📊 Segmentasi Kecamatan (PCA 2D Projection)")
    )

    return df_clustered, chart
