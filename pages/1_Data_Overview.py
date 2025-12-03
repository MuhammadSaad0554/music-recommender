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

st.subheader("Dataset Summary")

st.write("""
Contains 124,000 Spotify tracks.
Main interest in numerical audio features. 
""")

# Scrollable dataset preview
st.dataframe(spotify, use_container_width=True, height=350)

st.subheader("Long Tail of Popularity")

st.write("""
Most songs in the dataset are not highly popular. A very small fraction has high popularity,
while the majority fall in the lower ranges. This creates what is known as a long tail
distribution. Understanding this distribution helps explain why popularity was not used in
the recommendation model, and why a content-based approach was preferred.
""")

fig, ax = plt.subplots(figsize=(8, 4))
spotify["popularity"].hist(bins=50, ax=ax, color="skyblue", edgecolor="black")
ax.set_title("Distribution of Song Popularity")
ax.set_xlabel("Popularity Score")
ax.set_ylabel("Number of Songs")
st.pyplot(fig)

st.subheader("Top 20 Genres")

st.write("""
The dataset includes a wide variety of genre labels based on playlist metadata.
Here are the top twenty most common genres in the dataset.
""")

top_genres = spotify["track_genre"].value_counts().head(20)

fig2, ax2 = plt.subplots(figsize=(8, 6))
top_genres.plot(kind="bar", ax=ax2, color="lightgreen", edgecolor="black")
ax2.set_title("Top 20 Most Common Genres")
ax2.set_ylabel("Number of Songs")
ax2.set_xlabel("Genre")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig2)
