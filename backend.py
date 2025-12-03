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

    # Clean dataset
    spotify = spotify.dropna(subset=audio_features)
    spotify = spotify.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spotify[audio_features])

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    spotify["pca1"] = X_pca[:, 0]
    spotify["pca2"] = X_pca[:, 1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    spotify["cluster"] = kmeans.fit_predict(X_pca)

    return spotify, X_scaled


def hybrid_recommend(song_name, artist_name):
    spotify, feature_matrix = load_data()

    mask = (spotify["track_name"] == song_name) & (spotify["artists"] == artist_name)
    if not mask.any():
        return None

    idx = spotify[mask].index[0]

    genre = spotify.loc[idx, "track_genre"]
    cluster = spotify.loc[idx, "cluster"]

    # Same-genre, same-cluster subset
    subset = spotify[
        (spotify["track_genre"] == genre) &
        (spotify["cluster"] == cluster)
    ].copy()

    # Compute cosine similarity
    sims = cosine_similarity(
        feature_matrix[idx].reshape(1, -1),
        feature_matrix[subset.index]
    )[0]
    subset["similarity"] = sims

    # --- Categories ---

    # 1. SAME ARTIST (up to 5)
    same_artist = (
        subset[(subset["artists"] == artist_name) & (subset.index != idx)]
        .sort_values("similarity", ascending=False)
        .head(5)
    )

    # 2. SIMILAR POPULAR HITS (popularity ≥ 70)
    similar_popular = (
        subset[
            (subset["artists"] != artist_name) &
            (subset["popularity"] >= 70)
        ]
        .sort_values("similarity", ascending=False)
        .head(5)
    )

    # 3. HIDDEN GEMS (very obscure: popularity ≤ 10)
    hidden_gems = (
        subset[
            (subset["artists"] != artist_name) &
            (subset["popularity"] <= 10)
        ]
        .sort_values("similarity", ascending=False)
        .head(5)
    )

    return {
        "same_artist": same_artist[["track_name", "artists", "track_genre"]],
        "similar_popular": similar_popular[["track_name", "artists", "track_genre"]],
        "hidden_gems": hidden_gems[["track_name", "artists", "track_genre"]]
    }
