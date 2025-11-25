import streamlit as st
from backend import load_data, hybrid_recommend

st.title("Song Recommender")

# Load processed data
spotify, _ = load_data()

# Create dropdown list of songs
song_list = sorted(spotify["track_name"].unique())
song_name = st.selectbox("Select a song:", song_list)

# Filter artists for that song
artist_list = sorted(
    spotify[spotify["track_name"] == song_name]["artists"].unique()
)
artist_name = st.selectbox("Select the artist:", artist_list)

# Run recommender
if st.button("Recommend"):
    st.write("Top 10 Recommendations")

    recs = hybrid_recommend(song_name, artist_name, top_n=10)

    if recs.empty:
        st.warning("No recommendations found.")
    else:
        for _, row in recs.iterrows():
            track = row["track_name"]
            artist = row["artists"]

            yt = f"https://www.youtube.com/results?search_query={track}+{artist}".replace(" ", "+")
            sp = f"https://open.spotify.com/search/{track}%20{artist}".replace(" ", "%20")

            st.markdown(f"""
**{track} — {artist}**

[YouTube Search]({yt})  
[Spotify Search]({sp})

""")
