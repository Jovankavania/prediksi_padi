# src/clustering_app.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import altair as alt

def do_clustering(df, n_clusters=3):
    """
    Segmentasi kecamatan berdasarkan hasil prediksi produksi
    dan fitur-fitur pertanian terkait.

    Parameters
    ----------
    df : DataFrame
        DataFrame harus sudah mengandung kolom:
        ['Kecamatan','Luas Sawah','Luas Tanam','Luas Panen',
         'Rasio_Tanam','Intensitas_Sawah','Panen_x_Intensitas',
         'Tanam_x_Rasio','Prediksi Produksi']
    n_clusters : int
        Jumlah cluster yang diinginkan

    Returns
    -------
    df_clustered : DataFrame
        Data dengan tambahan kolom 'Cluster'
    chart : Altair Chart
        Visualisasi hasil clustering
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

    # --- normalisasi data biar KMeans tidak bias ke variabel besar ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- kmeans clustering ---
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    # --- tambahkan cluster ke dataframe ---
    df_clustered = df.copy()
    df_clustered["Cluster"] = labels

    # --- opsional: beri label kategori agar mudah dibaca ---
    df_clustered["Kategori"] = df_clustered["Cluster"].map({
        0: "Cluster 1 – Produksi Rendah",
        1: "Cluster 2 – Produksi Sedang",
        2: "Cluster 3 – Produksi Tinggi"
    })

    # --- buat visualisasi sederhana ---
    chart = (
        alt.Chart(df_clustered)
        .mark_circle(size=200)
        .encode(
            x=alt.X("Prediksi Produksi:Q", title="Produksi Prediksi"),
            y=alt.Y("Luas Panen:Q", title="Luas Panen"),
            color=alt.Color("Cluster:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                "Kecamatan","Prediksi Produksi","Luas Sawah","Luas Tanam",
                "Luas Panen","Rasio_Tanam","Intensitas_Sawah","Cluster","Kategori"
            ]
        )
        .properties(width=700, height=400,
                    title="📊 Segmentasi Kecamatan Berdasarkan Produksi & Lahan")
    )

    return df_clustered, chart
