import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.title("Pipeline Overview")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/MuhammadSaad0554/music-recommender/main/spotify.csv"
    df = pd.read_csv(url)
    df["track_name"] = df["track_name"].astype(str)
    df["artists"] = df["artists"].astype(str)
    return df

spotify = load_data()

st.subheader("Modeling Pipeline")

st.write("""
This section explains the steps used to build the music recommendation pipeline.
The system follows a content-based approach that relies entirely on audio features rather
than user behavior or popularity scores.
""")

st.header("Step 1: Data Cleaning")

st.write("""
The raw dataset contained many repeated observations because the same songs appeared
across multiple playlists. Duplicate entries were removed to ensure each song had one row.

Missing audio feature values were also removed to maintain consistency across computations.
""")

st.header("Step 2: Feature Scaling")

st.write("""
Audio features exist on different numeric scales. For example, loudness is measured in decibels
while danceability is measured from zero to one. To ensure equal weighting across features,
values were standardized using the StandardScaler.
""")

# Show scaled distribution example
audio_features = [
    "danceability", "energy", "valence", "acousticness",
    "speechiness", "instrumentalness", "liveness",
    "loudness", "tempo"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(spotify[audio_features])

st.write("""
Below is the distribution of one scaled feature as an example.
""")

fig1, ax1 = plt.subplots(figsize=(6, 3))
pd.Series(scaled[:, 0]).hist(ax=ax1, bins=50, color="purple", edgecolor="black")
ax1.set_title("Distribution of Scaled Danceability")
st.pyplot(fig1)

st.header("Step 3: PCA Dimensionality Reduction")

st.write("""
The model uses Principal Component Analysis (PCA) to reduce the nine audio features down to
two principal components. This reduces noise, improves clustering quality, and makes the data
visually interpretable in two dimensions.
""")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.4)
ax2.set_title("PCA Projection of Audio Features")
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
st.pyplot(fig2)

st.header("Step 4: KMeans Clustering")

st.write("""
After PCA, the model applies KMeans clustering to group songs into broad audio-based categories.
These clusters reflect broad musical tendencies such as energetic, acoustic, or mellow songs.

The clustering is not genre-based but derived entirely from audio patterns.
""")

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

fig3, ax3 = plt.subplots(figsize=(6, 4))
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, s=5, cmap="viridis", alpha=0.6)
ax3.set_title("KMeans Clusters in PCA Space")
ax3.set_xlabel("PCA 1")
ax3.set_ylabel("PCA 2")
st.pyplot(fig3)

st.header("Step 5: Cosine Similarity Based Recommendations")

st.write("""
Cosine similarity is used to compute how similar two songs are based on their audio feature vectors.
This measures the direction of the feature vectors rather than their magnitude, which is useful
when features vary widely in scale.

Final recommendations are filtered to ensure that songs come from the same playlist-derived genre.
This hybrid design improves musical coherence while still relying primarily on audio similarity.
""")

st.write("""
The next tab, Recommender, allows you to test the model using your own song selections.
""")
