import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import squarify

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
Contains 124,000 Spotify tracks.
Accessible on Kaggle.
Main interest in numerical audio features.
""")

st.dataframe(spotify, use_container_width=True, height=350)

st.subheader("Long Tail Challenge")

st.write("""
Small fraction of songs on any platform are popular.
Recommendation algorithms based on user choices (collaborative filtering) favor popular music.
Content based filtering can improve discovery with more equitable streaming opportunities.
""")

fig, ax = plt.subplots(figsize=(8, 4))
spotify["popularity"].plot(kind="density", ax=ax, color="purple", linewidth=2)
ax.set_title("Density Plot: Song Popularity")
ax.set_xlabel("Popularity Score")
ax.set_ylabel("")
ax.grid(False)
st.pyplot(fig)

st.subheader("Top 20 Genres")

top_genres = spotify["track_genre"].value_counts().head(20)

sizes = top_genres.values
labels = [f"{genre}\n{count}" for genre, count in top_genres.items()]

fig2, ax2 = plt.subplots(figsize=(10, 6))
squarify.plot(
    sizes=sizes,
    label=labels,
    alpha=0.85,
    color=plt.cm.viridis_r(range(len(sizes)))
)

plt.axis("off")
st.pyplot(fig2)
