import streamlit as st
from backend import load_data, recommend_three_categories

st.title("Song Recommender")

spotify, _ = load_data()

song_list = sorted(spotify["track_name"].unique())
song_name = st.selectbox("Select a song:", song_list)

artist_list = sorted(
    spotify[spotify["track_name"] == song_name]["artists"].unique()
)
artist_name = st.selectbox("Select the artist:", artist_list)

if st.button("Recommend"):
    st.write("Generating recommendations. Please wait.")

    same_artist, popular_hits, hidden_gems = recommend_three_categories(
        song_name, artist_name, top_n=3
    )

    if same_artist is None:
        st.warning("Song not found in the dataset.")
        st.stop()

    # SECTION 1: Same Artist Matches
    st.header("Same Artist Matches")
    if same_artist.empty:
        st.write("No similar tracks from the same artist were found.")
    else:
        for _, row in same_artist.iterrows():
            track = row["track_name"]
            artist = row["artists"]
            yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
            sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
            st.markdown(f"""
**{track} — {artist}**

YouTube Search  
{yt}

Spotify Search  
{sp}
""")

    # SECTION 2: Similar Popular Hits
    st.header("Similar Popular Hits")
    if popular_hits.empty:
        st.write("No popular high similarity tracks were found.")
    else:
        for _, row in popular_hits.iterrows():
            track = row["track_name"]
            artist = row["artists"]
            yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
            sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
            st.markdown(f"""
**{track} — {artist}**

YouTube Search  
{yt}

Spotify Search  
{sp}
""")

    # SECTION 3: Hidden Gems
    st.header("Hidden Gems")
    if hidden_gems.empty:
        st.write("No lower popularity tracks were found.")
    else:
        for _, row in hidden_gems.iterrows():
            track = row["track_name"]
            artist = row["artists"]
            yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
            sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")
            st.markdown(f"""
**{track} — {artist}**

YouTube Search  
{yt}

Spotify Search  
{sp}
""")
