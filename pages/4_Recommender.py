import streamlit as st
from backend import load_data, hybrid_recommend

st.title("Song Recommender")

# Load processed data
spotify, _ = load_data()

# Dropdown list of songs
song_list = sorted(spotify["track_name"].unique())
song_name = st.selectbox("Select a song:", song_list)

# Filter artists for that song
artist_list = sorted(
    spotify[spotify["track_name"] == song_name]["artists"].unique()
)
artist_name = st.selectbox("Select the artist:", artist_list)

# Run recommender
if st.button("Recommend"):
    st.write(f"Recommendations based on **{song_name} — {artist_name}**")

    recs = hybrid_recommend(song_name, artist_name)

    if recs is None:
        st.warning("Song not found in dataset.")
    else:

        st.header("From the Same Artist")
        same_artist = recs["same_artist"]
        if same_artist.empty:
            st.write("No same-artist songs found.")
        else:
            for _, row in same_artist.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
                st.markdown(f"""
                **{track} — {artist}**  
                [YouTube]({yt}) · [Spotify]({sp})
                """)

        st.header("Similar Popular Hits")
        similar_popular = recs["similar_popular"]
        if similar_popular.empty:
            st.write("No popular similar songs found.")
        else:
            for _, row in similar_popular.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
                st.markdown(f"""
                **{track} — {artist}**  
                [YouTube]({yt}) · [Spotify]({sp})
                """)

        st.header("Hidden Gems")
        hidden_gems = recs["hidden_gems"]
        if hidden_gems.empty:
            st.write("No hidden gems found.")
        else:
            for _, row in hidden_gems.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
                st.markdown(f"""
                **{track} — {artist}**  
                [YouTube]({yt}) · [Spotify]({sp})
                """)
