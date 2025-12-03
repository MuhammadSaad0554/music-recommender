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
Approximately 124,000 Spotify tracks from a Kaggle dataset.
Focus on numerical audio features to build a content based recommendation system.
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

st.subheader("Genres & Audio Features")

st.write("""
Genres ranked by their average across: **danceability**, 
**acousticness**, and **valence**. 
""")

# Compute averages per genre
genre_means = (
    spotify.groupby("track_genre")[["danceability", "acousticness", "valence"]]
    .mean()
    .reset_index()
)

# Sort genres for each feature
sorted_dance = genre_means.sort_values("danceability", ascending=False)[["track_genre"]]
sorted_acoustic = genre_means.sort_values("acousticness", ascending=False)[["track_genre"]]
sorted_valence = genre_means.sort_values("valence", ascending=False)[["track_genre"]]

# Display scrollable lists
st.write("### Genres: Danceability")
st.dataframe(sorted_dance, use_container_width=True, height=220)

st.write("### Genres: Acousticness")
st.dataframe(sorted_acoustic, use_container_width=True, height=220)

st.write("### Genres: Valence")
st.dataframe(sorted_valence, use_container_width=True, height=220)


