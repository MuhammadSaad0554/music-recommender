import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data Overview")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/MuhammadSaad0554/music-recommender/main/spotify.csv"
    df = pd.read_csv(url)
    df["track_name"] = df["track_name"].astype(str)
    df["artists"] = df["artists"].astype(str)
    return df

spotify = load_data()

st.subheader("Dataset")

st.write("""
This dataset contains approximately 124,000 Spotify tracks sourced from a publicly available Kaggle dataset.
The main focus of the project is on numerical audio features which are used to build a content based recommendation system.
""")

st.dataframe(spotify, use_container_width=True, height=350)

st.subheader("Long Tail Challenge")

st.write("""
Only a small fraction of songs are popular.
Collaborative filtering based on other users' preferences overemphasizes popular songs.
Content based approach helps with music discovery. 
""")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(spotify["popularity"], bins=range(0, 105, 5), color="purple", edgecolor="black")
ax.set_title("Distribution of Number of Songs by Popularity Score")
ax.set_xlabel("Popularity Score")
ax.set_ylabel("Number of Songs")
ax.grid(False)
st.pyplot(fig)

st.subheader("Audio Features & Genres")

st.write("""
Overview of top 5 genres by average danceability, acousticness and valence.
""")

# Compute mean features per genre
genre_means = (
    spotify.groupby("track_genre")[["danceability", "acousticness", "valence"]]
    .mean()
    .reset_index()
)

# Top 5 for each feature
top_dance = genre_means.nlargest(5, "danceability")
top_acoustic = genre_means.nlargest(5, "acousticness")
top_valence = genre_means.nlargest(5, "valence")

st.write("### Top 5 Genres by Danceability")
st.dataframe(top_dance.sort_values("danceability", ascending=False), 
             use_container_width=True, height=180)

st.write("### Top 5 Genres by Acousticness")
st.dataframe(top_acoustic.sort_values("acousticness", ascending=False), 
             use_container_width=True, height=180)

st.write("### Top 5 Genres by Valence (Musical Positivity)")
st.dataframe(top_valence.sort_values("valence", ascending=False), 
             use_container_width=True, height=180)

