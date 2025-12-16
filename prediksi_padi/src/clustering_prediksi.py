# src/clustering_app.py

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

    # 1) Simpan cluster angka dengan nama yang jelas
    df_clustered["Cluster_Angka"] = labels

    # (Kalau kamu masih butuh nama "Cluster" untuk peta/kompatibilitas, tetap simpan juga)
    df_clustered["Cluster"] = df_clustered["Cluster_Angka"]

    # 2) Label bisnis sesuai interpretasimu
    label_map = {
        0: "Unggul",
        1: "Kecil",
        2: "Sedang"
    }
    df_clustered["Cluster_Label"] = df_clustered["Cluster_Angka"].map(label_map)

    # 3) Urutan tampil (opsional) biar konsisten untuk BI report
    order_map = {"Unggul": 1, "Sedang": 2, "Kecil": 3}
    df_clustered["Urutan_Klaster"] = df_clustered["Cluster_Label"].map(order_map)

    # PCA untuk visualisasi
    pca = PCA(n_components=2, random_state=42)
