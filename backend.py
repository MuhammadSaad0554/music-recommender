
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    file_path = "/Users/muhammadsaad/Desktop/Georgetown/Semester_3/DSIII/Project/Data/spotify.csv"
    spotify = pd.read_csv(file_path)

    audio_features = [
        "danceability", "energy", "valence", "acousticness",
        "speechiness", "instrumentalness", "liveness",
        "loudness", "tempo"
    ]

    spotify = spotify.dropna(subset=audio_features)
    spotify = spotify.drop_duplicates(subset=["track_name", "artists"]).copy()
    spotify = spotify.reset_index(drop=True)

    spotify["track_name"] = spotify["track_name"].str.title()
    spotify["artists"] = spotify["artists"].str.title()
    spotify["track_genre"] = spotify["track_genre"].fillna("").str.lower()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spotify[audio_features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    spotify["pca1"] = X_pca[:, 0]
    spotify["pca2"] = X_pca[:, 1]

    kmeans = KMeans(n_clusters=30, random_state=42)
    spotify["cluster"] = kmeans.fit_predict(X_pca)

    return spotify, X_scaled

spotify, feature_matrix = load_data()

def hybrid_recommend(song_name, artist_name=None, top_n=10):
    if artist_name:
        match = spotify[
            (spotify["track_name"] == song_name) &
            (spotify["artists"] == artist_name)
        ]
    else:
        match = spotify[spotify["track_name"] == song_name]

    if match.empty:
        return pd.DataFrame()

    idx = match.index[0]

    song_genre = spotify.loc[idx, "track_genre"]
    song_cluster = spotify.loc[idx, "cluster"]

    genre_subset = spotify[spotify["track_genre"] == song_genre]
    cluster_subset = genre_subset[genre_subset["cluster"] == song_cluster]

    if cluster_subset.shape[0] < top_n + 1:
        cluster_subset = genre_subset

    subset_idx = cluster_subset.index.tolist()

    sims = cosine_similarity(
        feature_matrix[idx].reshape(1, -1),
        feature_matrix[subset_idx]
    )[0]

    top_indices = sims.argsort()[::-1][1:top_n+1]

    recs = cluster_subset.iloc[top_indices][[
        "track_name", "artists", "track_genre", "cluster"
    ]]

    return recs.drop_duplicates(subset=["track_name", "artists"])
