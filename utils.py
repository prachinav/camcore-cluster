import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from folium import DivIcon
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import folium
import os

import matplotlib
matplotlib.use('Agg')

def plot_cluster_diagnostics(df, upload_dir):
    optimal_k = 5
    df = df[df['scenario'] == 'future'].copy()
    features = df.filter(like="bio").dropna()

    # Remove highly correlated features (correlation > 0.9)
    corr_matrix = features.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    features_filtered = features.drop(columns=to_drop)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    silhouette_scores, wss, db_indexes = [], [], []

    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, n_init=25, random_state=42).fit(scaled)
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(scaled, labels))
        wss.append(kmeans.inertia_)
        db_indexes.append(davies_bouldin_score(scaled, labels))

    best_silhouette_k = int(np.argmax(silhouette_scores) + 2)
    best_db_k = int(np.argmin(db_indexes) + 2)
    knee = KneeLocator(list(range(2, 21)), wss, curve="convex", direction="decreasing")
    elbow_k = knee.elbow

    os.makedirs(upload_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # Silhouette Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(x=range(2, 21), y=silhouette_scores, ax=ax, color='steelblue')
    ax.axvline(best_silhouette_k, color='green', linestyle='--', label=f'Silhouette max at k={best_silhouette_k}')
    ax.set_title("Silhouette Method")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    silhouette_path = os.path.join(upload_dir, 'silhouette_method.png')
    fig.tight_layout()
    fig.savefig(silhouette_path)
    plt.close(fig)

    # Elbow Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(x=range(2, 21), y=wss, ax=ax, color='steelblue')
    ax.axvline(elbow_k, color='green', linestyle='--', label=f'Elbow at k={elbow_k}')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WSS (Within-Cluster Sum of Squares)")
    ax.legend()
    elbow_path = os.path.join(upload_dir, 'elbow_method.png')
    fig.tight_layout()
    fig.savefig(elbow_path)
    plt.close(fig)

    # Davies–Bouldin Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.lineplot(x=range(2, 21), y=db_indexes, ax=ax, color='steelblue')
    ax.axvline(best_db_k, color='green', linestyle='--', label=f'DB min at k={best_db_k}')
    ax.set_title("Davies–Bouldin Index")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("DB Index")
    ax.legend()
    db_path = os.path.join(upload_dir, 'davies_bouldin_index.png')
    fig.tight_layout()
    fig.savefig(db_path)
    plt.close(fig)

    return silhouette_path, elbow_path, db_path


def run_clustering_pipeline(csv_path, k, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df['region'].astype(str).str.startswith("MM")]

    df_future = df[df['scenario'] == "future"].copy()
    future_features = df_future.iloc[:, 5:43].dropna()
    future_bio = future_features.filter(regex='^bio')
    scaler = StandardScaler()
    f_scaled = scaler.fit_transform(future_bio)

    # KMeans
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=123).fit(f_scaled)
    df_future = df_future.loc[future_bio.index].copy()
    df_future['cluster'] = kmeans.labels_.astype(str)
    centroids = kmeans.cluster_centers_

    df_past = df[df['scenario'] == "past"].copy()
    past_features = df_past.iloc[:, 5:43].dropna()
    past_bio = past_features.filter(regex='^bio')
    f_scaled_past = scaler.transform(past_bio)
    df_past = df_past.loc[past_bio.index].copy()

    def assign_cluster(x): return str(np.argmin(np.linalg.norm(centroids - x, axis=1)))
    df_past['cluster'] = [assign_cluster(row) for row in f_scaled_past]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(f_scaled)
    df_future['pca1'], df_future['pca2'] = pca_result[:, 0], pca_result[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_future, x='pca1', y='pca2', hue='cluster', palette='viridis', ax=ax)
    ax.set_title("PCA of Future Clusters")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_clusters.png"))
    plt.close(fig)

    def plot_map(df_map, title, filename):
        m = folium.Map(location=[df_map['lat'].mean(), df_map['lon'].mean()], zoom_start=6)
        clusters = sorted(df_map['cluster'].unique())
        colors = sns.color_palette("hsv", len(clusters)).as_hex()
        color_map = {str(c): colors[i] for i, c in enumerate(clusters)}

        for _, row in df_map.iterrows():
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=6,
                color=color_map[row['cluster']],
                fill=True,
                fill_opacity=0.7,
            ).add_to(m)

            folium.map.Marker(
                [row['lat'], row['lon']],
                icon=DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 10pt; color: black; background: white; padding: 1px 2px;">{row["cluster"]}</div>'
                )
            ).add_to(m)

        legend = '<div style="position: fixed; bottom: 30px; left: 30px; width: 160px; background: white; border:2px solid grey; z-index:9999; padding: 10px;">'
        legend += f"<b>{title}</b><br>"
        for c, col in color_map.items():
            legend += f'<i style="background:{col};width:12px;height:12px;display:inline-block;margin-right:5px;"></i>Cluster {c}<br>'
        legend += '</div>'
        m.get_root().html.add_child(folium.Element(legend))
        m.save(os.path.join(output_dir, filename))

    def plot_transition_map(future_df, past_df, filename):
        merged = pd.merge(
            future_df[['id', 'lat', 'lon', 'cluster']].rename(columns={'cluster': 'future'}),
            past_df[['id', 'cluster']].rename(columns={'cluster': 'past'}),
            on='id'
        )
        changed = merged[merged['future'] != merged['past']]
        m = folium.Map(location=[changed['lat'].mean(), changed['lon'].mean()], zoom_start=6)

        for _, row in changed.iterrows():
            folium.CircleMarker(
                location=(row['lat'], row['lon']),
                radius=6,
                color='red',
                fill=True,
                fill_opacity=0.8
            ).add_to(m)

            folium.map.Marker(
                [row['lat'], row['lon']],
                icon=DivIcon(
                    icon_size=(150, 36),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 10pt; color: red; background: white; padding: 1px 2px;">{row["id"]}</div>'
                )
            ).add_to(m)

        legend = """
        <div style="position: fixed; bottom: 30px; left: 30px; width: 180px; background: white; border:2px solid red; z-index:9999; padding: 10px;">
        <b>Changed Clusters</b><br>
        <i style="background:red;width:12px;height:12px;display:inline-block;margin-right:5px;"></i>Cluster Change
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend))
        m.save(os.path.join(output_dir, filename))

    # ---- Plot maps ----
    plot_map(df_future, "Future Clusters", "map_future.html")
    plot_map(df_past, "Past Clusters", "map_past.html")
    plot_transition_map(df_future, df_past, "map_transition.html")

    return {
        "pca_img": "static/output/pca_clusters.png",
        "map_future": "static/output/map_future.html",
        "map_past": "static/output/map_past.html",
        "map_transition": "static/output/map_transition.html"
    }