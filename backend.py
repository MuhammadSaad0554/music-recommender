import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    url = "https://raw.githubusercontent.com/MuhammadSaad0554/music-recommender/main/spotify.csv"
    spotify = pd.read_csv(url)

    spotify["track_name"] = spotify["track_name"].astype(str)
    spotify["artists"] = spotify["artists"].astype(str)

    audio_features = [
        "danceability", "energy", "valence", "acousticness",
        "speechiness", "instrumentalness", "liveness",
        "loudness", "tempo"
    ]

    spotify = spotify.dropna(subset=audio_features)
    spotify = spotify.drop_duplicates(subset=["track_name", "artists"]).copy()
    spotify = spotify.reset_index(drop=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spotify[audio_features])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    spotify["pca1"] = X_pca[:, 0]
    spotify["pca2"] = X_pca[:, 1]

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    spotify["cluster"] = clusters

    return spotify, X_scaled


def recommend_three_categories(song_name, artist_name, top_n=3):
    spotify, feature_matrix = load_data()

    mask = (spotify["track_name"] == song_name) & (spotify["artists"] == artist_name)
    if not mask.any():
        return None, None, None

    idx = spotify[mask].index[0]

    target_genre = spotify.loc[idx, "track_genre"]
    target_cluster = spotify.loc[idx, "cluster"]
    target_vec = feature_matrix[idx].reshape(1, -1)

    same_group = spotify[
        (spotify["track_genre"] == target_genre) &
        (spotify["cluster"] == target_cluster)
    ]

    sims = cosine_similarity(target_vec, feature_matrix[same_group.index])[0]
    same_group = same_group.assign(similarity=sims)

    # same artist matches
    same_artist = same_group[
        same_group["artists"] == artist_name
    ].sort_values("similarity", ascending=False)

    same_artist = same_artist[same_artist.index != idx].head(top_n)

    # similar popular hits
    pop_threshold = same_group["popularity"].quantile(0.75)

    popular_hits = same_group[
        (same_group["artists"] != artist_name) &
        (same_group["popularity"] >= pop_threshold)
    ].sort_values("similarity", ascending=False).head(top_n)

    # hidden gems with improved logic
    gem_threshold = same_group["popularity"].quantile(0.30)

    hidden_gems = same_group[
        (same_group["artists"] != artist_name) &
        (same_group["popularity"] <= gem_threshold)
    ]

    hidden_gems = hidden_gems[
        hidden_gems["similarity"] > 0.6
    ].sort_values("similarity", ascending=False).head(top_n)

    return same_artist, popular_hits, hidden_gems
