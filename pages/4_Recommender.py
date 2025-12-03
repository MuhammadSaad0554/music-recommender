import streamlit as st
from backend import load_data, recommend_three_categories

st.title("Song Recommender")

# Load data
spotify, _ = load_data()

# Dropdown for song
song_list = sorted(spotify["track_name"].unique())
song_name = st.selectbox("Select a song:", song_list)

# Dropdown for artist
artist_list = sorted(
    spotify[spotify["track_name"] == song_name]["artists"].unique()
)
artist_name = st.selectbox("Select the artist:", artist_list)

# Generate recommendations
if st.button("Recommend"):
    same_artist, popular_hits, hidden_gems = recommend_three_categories(song_name, artist_name)

    if same_artist is None:
        st.warning("Song not found in the dataset.")
    else:

        # Section 1: Same Artist
        st.header("More from this Artist")
        if same_artist.empty:
            st.write("No other songs by this artist in this cluster and genre.")
        else:
            for _, row in same_artist.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")

                st.markdown(f"""
**{track} — {artist}**  
[YouTube Search]({yt})  
[Spotify Search]({sp})
                """)

        # Section 2: Popular Hits
        st.header("Similar Popular Hits")
        if popular_hits.empty:
            st.write("No popular close matches.")
        else:
            for _, row in popular_hits.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")

                st.markdown(f"""
**{track} — {artist}**  
[YouTube Search]({yt})  
[Spotify Search]({sp})
                """)

        # Section 3: Hidden Gems
        st.header("Hidden Gems")
        if hidden_gems.empty:
            st.write("No hidden gems found within this cluster and genre.")
        else:
            for _, row in hidden_gems.iterrows():
                track = row["track_name"]
                artist = row["artists"]
                yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
                sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")

                st.markdown(f"""
**{track} — {artist}**  
[YouTube Search]({yt})  
[Spotify Search]({sp})
                """)
