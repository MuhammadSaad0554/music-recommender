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
This section explains the steps used to build the recommendation system.
The model is fully content-based and relies on audio features rather than popularity or user data.
""")

st.header("Step 1: Data Cleaning")

st.write("""
Duplicate tracks were removed because the same songs appeared across multiple playlists.
Rows missing audio features were dropped to maintain consistency in scaling and similarity computation.
""")

st.header("Step 2: Feature Scaling")

st.write("""
Audio features such as danceability, valence, tempo, and loudness exist on different numeric scales.
StandardScaler was applied so that all features are standardized before PCA and similarity.
""")

audio_features = [
    "danceability", "energy", "valence", "acousticness",
    "speechiness", "instrumentalness", "liveness",
    "loudness", "tempo"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(spotify[audio_features])

fig1, ax1 = plt.subplots(figsize=(6, 3))
pd.Series(scaled[:, 0]).hist(ax=ax1, bins=40, color="purple", edgecolor="black")
ax1.set_title("Scaled Danceability Distribution")
st.pyplot(fig1)

st.header("Step 3: PCA for Dimensionality Reduction")

st.write("""
PCA reduces the nine audio features into two principal components. This helps reduce noise,
improves clustering quality, and allows the system to visualize songs in a two-dimensional space.
""")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.scatter(X_pca[:, 0], X_pca[:, 1], s=5, alpha=0.4)
ax2.set_title("PCA Projection of Songs")
st.pyplot(fig2)

st.header("Step 4: KMeans Clustering")

st.write("""
KMeans groups songs based on their PCA-transformed audio features. These clusters reflect 
general musical tendencies such as high-energy, acoustic, or mellow tracks.
""")

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=5, alpha=0.6)
ax3.set_title("KMeans Clusters in PCA Space")
st.pyplot(fig3)

st.header("Step 5: Cosine Similarity")

st.write("""
Cosine similarity computes the closeness between songs based on audio feature vectors.
The model then filters results by genre to improve relevance and prevent mismatched recommendations.
""")

st.write("You can now try generating recommendations in the Recommender tab.")
