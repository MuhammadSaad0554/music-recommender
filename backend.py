import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    # Load CSV from GitHub raw link
    url = "https://raw.githubusercontent.com/MuhammadSaad0554/music-recommender/main/spotify.csv"
    spotify = pd.read_csv(url)

    # Ensure strings
    spotify["track_name"] = spotify["track_name"].astype(str)
    spotify["artists"] = spotify["artists"].astype(str)

    # Audio feature columns
    audio_features = [
        "danceability", "energy", "valence", "acousticness",
        "speechiness", "instrumentalness", "liveness",
        "loudness", "tempo"
    ]

    # Drop missing values for audio features
    spotify = spotify.dropna(subset=audio_features)

    # Drop duplicates
    spotify = spotify.drop_duplicates(subset=["track_name", "artists"]).copy()

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(spotify[audio_features])

    # PCA to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    spotify["pca1"] = X_pca[:, 0]
    spotify["pca2"] = X_pca[:, 1]

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    spotify["cluster"] = clusters

    # Return dataframe and scaled feature matrix
    feature_matrix = X_scaled
    return spotify, feature_matrix


def hybrid_recommend(song_name, artist_name, top_n=10):
    # Reload processed data each time
    spotify, feature_matrix = load_data()

    # Identify matching song and artist
    mask = (spotify["track_name"] == song_name) & (spotify["artists"] == artist_name)
    if not mask.any():
        return pd.DataFrame()

    idx = spotify[mask].index[0]

    # Use genre if available
    if "track_genre" in spotify.columns:
        genre = spotify.loc[idx, "track_genre"]
        genre_subset = spotify[spotify["track_genre"] == genre]
    else:
        genre_subset = spotify

    subset_idx = genre_subset.index.tolist()

    # Compute similarities only within the genre
    sims = cosine_similarity(
        feature_matrix[idx].reshape(1, -1),
        feature_matrix[subset_idx]
    )[0]

    # Sort high to low similarity
    sorted_idx = sims.argsort()[::-1]

    # Skip the song itself
    sorted_idx = sorted_idx[1 : top_n + 1]

    # Map back to dataframe indices
    rec_rows = [subset_idx[i] for i in sorted_idx]

    return spotify.loc[rec_rows, ["track_name", "artists", "track_genre"]]
