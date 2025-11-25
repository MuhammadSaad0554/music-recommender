import streamlit as st
from backend import spotify, hybrid_recommend

# Convert columns to string to avoid sorting issues
spotify["track_name"] = spotify["track_name"].astype(str)
spotify["artists"] = spotify["artists"].astype(str)
spotify["track_genre"] = spotify["track_genre"].astype(str)

# Remove invalid rows
spotify = spotify[spotify["track_name"].str.lower() != "nan"]
spotify = spotify[spotify["artists"].str.lower() != "nan"]

st.title("Song Recommender")

# Text input (instead of dropdown)
song_name = st.text_input("Enter the song name (example: In The End)")
artist_name = st.text_input("Enter the artist name (optional, example: Linkin Park)")

if st.button("Recommend"):
    if song_name.strip() == "":
        st.error("Please enter a song name.")
    else:
        st.write("Top 10 Recommendations")

        # Normalize capitalization
        input_song = song_name.strip().title()
        input_artist = artist_name.strip().title() if artist_name.strip() else None

        recs = hybrid_recommend(input_song, input_artist, top_n=10)

        if recs.empty:
            st.warning("No recommendations found. Check spelling and try again.")
        else:
            for _, row in recs.iterrows():
                track = row["track_name"]
                artist = row["artists"]

                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")

                st.write(f"{track} by {artist}")
                st.write(f"YouTube: {yt}")
                st.write(f"Spotify: {sp}")
                st.write("")  # Line break