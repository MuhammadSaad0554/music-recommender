import streamlit as st

st.title("Project Overview")

st.subheader("Dataset")
st.write("""
The dataset used in this project contains Spotify tracks curated by a Kaggle contributor.
It includes more than one hundred thousand songs, along with metadata such as track name,
artists, album, and playlist-derived genre labels.

The primary focus of this project is on the numerical audio features, including:
danceability, energy, valence, acousticness, speechiness, instrumentalness,
liveness, loudness, and tempo.

These audio features are commonly used in music information retrieval research and
enable similarity-based recommendation systems.
""")

st.subheader("Model Overview")
st.write("""
This project uses a content-based recommendation approach built around audio
features, unsupervised learning, and cosine similarity. The main steps are:

1. Data Preprocessing  
   The raw dataset contained many duplicates, since the same song appeared in multiple playlists.
   Duplicates were removed, and track names were cleaned to ensure consistent string formatting.

2. Feature Scaling  
   Audio features have different numeric ranges. Standardizing them ensures that no
   single feature dominates the similarity computation.

3. Dimensionality Reduction with PCA  
   PCA reduces the nine audio features into two principal components. This helps remove noise,
   increases cluster separation, and improves the quality of downstream methods.

4. Clustering with KMeans  
   Songs are grouped into broad musical categories. A higher value of k (currently set to 25)
   improves recommendation quality by producing more granular clusters.

5. Cosine Similarity  
   Recommendations are generated based on cosine similarity over the scaled feature space.
   Cosine similarity focuses on the direction of a feature vector rather than magnitude, which
   is appropriate for comparing musical characteristics.

6. Similarity Constrained by Genre  
   To avoid cross-genre confusion in the recommendations, similarity is computed only among songs
   that share the same playlist-based genre label. This hybrid approach improves recommendation
   relevance but also introduces a limitation: genre labels themselves are noisy and do not
   perfectly separate musical styles.

Overall, this system implements a transparent and interpretable example of a
content-based music recommendation model.
""")
