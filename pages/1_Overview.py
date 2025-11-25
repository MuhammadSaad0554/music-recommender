import streamlit as st

st.title("Project Overview")

st.subheader("Dataset")
st.write("""
The dataset used in this project contains Spotify tracks based on a publicly available Kaggle dataset.
It includes more than one hundred thousand songs, along with metadata such as track name,
artists, album, and playlist-derived genre labels.

My main focus was on the numerical audio features, including:
danceability, energy, valence, acousticness, speechiness, instrumentalness,
liveness, loudness, and tempo.

These audio features are commonly used in music information retrieval research and
used in similarity-based recommendation systems.
""")

st.subheader("Model Overview")
st.write("""
The main steps are:

1. Data Preprocessing  
   The raw dataset contained many duplicates, as same songs appeared in multiple playlists.
   Duplicates were removed, and track names were cleaned to ensure consistent string formatting.

2. Feature Scaling  
   Audio features have different numeric ranges, so had to standardize. 

3. Dimensionality Reduction with PCA  
   Used PCA to reduce the nine audio features into two principal components. This helped remove noise and
   improve cluster quality.

4. Clustering with KMeans  
   Songs are clustered into broad musical categories. My cluster evaluation is not robust and based on trial and error.
   Found that a higher value of k (currently set to 25) improved the final recommendations as opposed to a lower one. 

5. Cosine Similarity  
   Recommendations are generated based on cosine similarity.
   Wanted to focus more on direction than magnitude.

6. Similarity Based on Genre  
   Similarity is computed only among songs from similar genre.
   This hybrid approach improves recommendation
   relevance but also introduces a limitation: genres are not neatly split and can cross over. 

""")
