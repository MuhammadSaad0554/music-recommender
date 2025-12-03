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
The model is fully content-based and uses audio features rather than user behavior.
""")

# Prepare audio features
audio_features = [
    "danceability", "energy", "valence", "acousticness",
    "speechiness", "instrumentalness", "liveness",
    "loudness", "tempo"
]

# Step 1: Cleaning
st.header("Step 1: Data Cleaning")
st.write("""
Duplicate tracks were removed because the same songs appear across many playlists.
Rows missing audio features were dropped to keep scaling and similarity consistent.
""")

# Step 2: Feature Scaling
st.header("Step 2: Feature Scaling")
st.write("""
Audio features vary widely in scale (e.g., tempo vs acousticness). 
StandardScaler ensures all features contribute evenly before PCA and clustering.
""")

scaler = StandardScaler()
scaled = scaler.fit_transform(spotify[audio_features])

# Step 3: PCA
st.header("Step 3: PCA for Dimensionality Reduction")
st.write("""
PCA reduces the nine audio features into two principal components. 
This simplifies clustering and helps reduce noise in the feature space.
""")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled)

# Step 4: Elbow plot for choosing k
st.header("Step 4: Choosing Number of Clusters")
st.write("""
To choose the number of clusters, the Sum of Squared Errors (SSE) was calculated for 
different values of k. A clear bend at **k = 4** suggests four meaningful clusters.
""")

sse = []
k_values = list(range(2, 11))

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42).fit(X_pca)
    sse.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4))
ax_elbow.plot(k_values, sse, marker="o")
ax_elbow.set_xlabel("Number of Clusters (k)")
ax_elbow.set_ylabel("SSE")
ax_elbow.set_title("Elbow Plot for Choosing k")
st.pyplot(fig_elbow)

# Step 5: Fit final kmeans model
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)

spotify["cluster"] = clusters

# Step 6: Top 5 genres in each cluster
st.header("Cluster Interpretation")
st.write("""
Each cluster represents a broad musical tendency based on audio features. 
The tables below show the **top 5 genres** for each cluster and their 
interpretation based on acoustic and rhythmic patterns.
""")

genre_counts = (
    spotify.groupby(["cluster", "track_genre"])
    .size()
    .reset_index(name="count")
)

for c in range(4):
    st.subheader(f"Cluster {c}")

    # get top 5 genres
    top5 = (
        genre_counts[genre_counts["cluster"] == c]
        .sort_values("count", ascending=False)
        .head(5)
    )
    st.dataframe(top5, use_container_width=True)

    # Add descriptive text
    if c == 0:
        st.write("""
        **Description:**  
        Acoustic, chill, folk-inspired, and mellow tracks. 
        Contains genres such as acoustic, chill, romance, tango, cantopop, and study music.
        Represents warm, relaxed, and instrument-forward songs.
        """)
    elif c == 1:
        st.write("""
        **Description:**  
        Ambient, classical, and sleep music with very low energy and high acousticness. 
        Includes classical, piano, opera, ambient, and new-age.
        Reflects soothing, slow, atmospheric tracks.
        """)
    elif c == 2:
        st.write("""
        **Description:**  
        High-energy electronic and heavy styles such as metal, trance, hardstyle, drum-and-bass.
        Represents loud, fast, intense, and aggressive tracks.
        """)
    elif c == 3:
        st.write("""
        **Description:**  
        Danceable, rhythmic global genres such as salsa, forró, dancehall, afrobeat, disco, and house.
        Represents upbeat, lively, groove-oriented music.
        """)

# Step 7: Recommendation Logic
st.header("How Recommendations Are Generated")

st.write("""
The system provides **three complementary types of recommendations** for each selected song:

1. **Same Artist Recommendations**  
   Songs by the same artist, within the same cluster and genre, ranked by cosine similarity.  
   These reflect artistic continuity.

2. **Similar Popular Hits**  
   Songs from the same cluster and genre but by different artists.  
   Ranked by cosine similarity but filtered for high popularity.  
   These capture mainstream, widely-liked tracks.

3. **Deep Cuts (Less Popular Songs)**  
   Same cluster and genre, but filtered for lower popularity.  
   These highlight lesser-known but musically similar songs.

Cosine similarity ensures that recommendations are driven by the **shape** of the audio feature vector, 
while clustering and genre filtering control for style, energy, and musical context.
""")

st.write("You can now try the recommender in the sidebar.")
