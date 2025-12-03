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
Only a small fraction of songs are highly popular while most songs fall into the lower popularity range.
This creates what is known as a long tail distribution and explains why collaborative filtering would overemphasize popular songs.
A content based approach helps uncover tracks that receive fewer plays but may still match a user's musical taste.
""")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(spotify["popularity"], bins=range(0, 105, 5), color="purple", edgecolor="black")
ax.set_title("Distribution of Number of Songs by Popularity Score")
ax.set_xlabel("Popularity Score")
ax.set_ylabel("Number of Songs")
ax.grid(False)
st.pyplot(fig)

st.subheader("Genre Availability")

st.write("""
See below ten genres with the highest number of songs and the ten genres with the lowest number of songs in the dataset.
""")

genre_counts = spotify["track_genre"].value_counts()

top10 = genre_counts.head(10).sort_index().reset_index()
top10.columns = ["Genre", "Song Count"]

bottom10 = genre_counts.tail(10).sort_index().reset_index()
bottom10.columns = ["Genre", "Song Count"]

st.write("Top Ten Genres by Song Count")
st.dataframe(top10, use_container_width=True, height=250)

st.write("Bottom Ten Genres by Song Count")
st.dataframe(bottom10, use_container_width=True, height=250)
