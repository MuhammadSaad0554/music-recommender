import streamlit as st

st.title("Project Overview")

st.subheader("Dataset")
st.write("""
The dataset used in this project contains Spotify tracks.
It is curated by the Kaggle user Maharshi Pandya and is publicly available.
It includes 114,000 songs, including metadata such as track name, artists, album, and genre.
I was interested in numerical audio features, including: danceability, energy, valence, acousticness,
speechiness, instrumentalness, liveness, loudness, and tempo.

These audio features can be used to developed similarity-based music retrieval systems.

""")

st.subheader("Model Overview")
st.write("""
The major components are:

1. **Data Preprocessing**
   Basic preprocessing and data cleaning. There were a lot of song duplicates, which had to be removed. 
   Track names had to coded conistently as strings. 

2. **Feature Scaling**
   The numerical audio features were on different scales, so had to be standardized. 

3. **Dimensionality Reduction with PCA**
   Used PCA to compresses the nine audio features into two
   principal components. Done to improve clustering quality and is a standard practice in such systems. 

4. **Clustering with KMeans**
   Clustered using KMeans into broad musical categories. My cluster evaluation right is not very robust. 
   It was based on trial and error. I found that a higher cluster number (right now set at 25) works better
   in terms of recommendations. 

5. **Cosine Similarity**
   Recommended songs are based on cosine similarity. Wanted to capture more of direction than difference in magnitude.

6. **Similarity Constrained by Genre**
   Added this so that to algorithm restricts similarity
   calculations to songs with same genre. This a hybrid approach but also a limitation. 
   Genres are not neatly separated and often cross over. 
   Limiting by genre reduces the range of recommendations. 

This is an example of **content-based clustering**. It develops recommendations exclusively based on audio-features. 
Alternative would be **collaborative clustering**, which would make recommendations based on preferences of other users. 
""")

