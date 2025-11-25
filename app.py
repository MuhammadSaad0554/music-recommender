import streamlit as st

st.set_page_config(
    page_title="Introduction",
)

st.title("Introduction")

st.write("""
This application demonstrates a simple content-based music recommendation system built 
using Spotify audio features and unsupervised learning techniques. The pipeline preprocess the data, 
scales numerical audio features, reduces dimensions using PCA, 
clusters songs based on K-means, and finall generates recommendations
using cosine similarity within largely similar genres.

Before you jump into the recommender, **please carefully read the User Instructions** from the sidebar.
""")
