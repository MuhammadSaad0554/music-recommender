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
Recommendations are exclusively content based using Spotify audio features.
""")

# Step 1: Data Cleaning
st.header("Step 1: Data Cleaning")

st.write("""
Duplicate tracks removed. 
Tracks with missing audio features dropped.
Consistent labelling. 
""")

# Step 2: Feature Scaling
st.header("Step 2: Feature Scaling")

st.write("""
Audio features were on different numeric scales. 
StandardScaler used. 
""")

audio_features = [
    "danceability", "energy", "valence", "acousticness",
    "speechiness", "instrumentalness", "liveness",
    "loudness", "tempo"
]

scaler = StandardScaler()
scaled = scaler.fit_transform(spotify[audio_features])

# Step 3: PCA
st.header("Step 3: PCA Dimensionality Reduction")

st.write("""
The nine audio features were reduced to two principal components using PCA.
Reduced noise and improved cluster separation. 
""")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled)

# Step 4: KMeans Clustering
st.header("Step 4: KMeans Clustering")

st.write("""
Used KMeans clustering.
Used the SSE based Elbow Plot to look at the ideal number of clusters, which were 4.
""")

sse = []
k_values = range(2, 13)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_pca)
    sse.append(km.inertia_)

fig_sse, ax_sse = plt.subplots(figsize=(6, 4))
ax_sse.plot(k_values, sse, marker='o')
ax_sse.set_title("SSE versus Number of Clusters")
ax_sse.set_xlabel("Number of Clusters k")
ax_sse.set_ylabel("SSE")
st.pyplot(fig_sse)

st.write("""
Based on the elbow method, the model uses four clusters which are labeled A, B, C, and D.
The table below shows the most frequent genres in each cluster.
""")

# Fit final cluster model
kmeans = KMeans(n_clusters=4, random_state=42)
spotify["cluster"] = kmeans.fit_predict(X_pca)

# Mapping from numeric cluster to label
cluster_labels = {0: "A", 1: "B", 2: "C", 3: "D"}
spotify["cluster_label"] = spotify["cluster"].map(cluster_labels)

top_genres_by_cluster = (
    spotify.groupby(["cluster_label", "track_genre"])
    .size()
    .reset_index(name="count")
    .sort_values(["cluster_label", "count"], ascending=[True, False])
)

# Cluster Interpretation
st.header("Cluster Interpretation")

for lbl in ["A", "B", "C", "D"]:
    st.subheader(f"Cluster {lbl} Top Genres")
    subset = top_genres_by_cluster[top_genres_by_cluster["cluster_label"] == lbl]
    st.dataframe(
        subset[["track_genre"]].head(10),
        use_container_width=True,
        height=240
    )

# Step 5: Cosine Similarity
st.header("Step 5: Cosine Similarity")

st.write("""
Cosine similarity used to measure how close two songs are in terms of their audio feature vectors.
""")

# Step 6: Recommendation Logic
st.header("Step 6: How Recommendations Are Generated")

st.write("""
The system produces three categories of recommendations. 
All three categories rank songs using cosine similarity. 
The difference between them lies in the filtering rules applied before ranking.

1. Same Artist Matches  
   These songs come from the same genre, the same cluster, and the same artist as the selected song.
   They are then ranked by cosine similarity. The top three are returned.

2. Similar Popular Hits  
   These songs come from the same genre and the same cluster as the selected track. 
   They must be from different artists and have higher popularity. 
   They are ranked by cosine similarity, and the top three are returned.

3. Hidden Gems  
   These songs come from the same genre and the same cluster as the selected track. 
   They must be from different artists, but they have lower popularity.
   They are also ranked by cosine similarity, and the top three are returned.

This design balances musical similarity with variety by offering familiar matches, 
mainstream options, and discovery oriented recommendations.
""")

st.write("You can now try generating recommendations in the Recommender tab.")
