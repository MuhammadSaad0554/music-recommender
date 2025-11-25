import streamlit as st

st.title("User Instructions")

st.subheader("How to Use the Recommender")

st.write("""
This application recommends songs based on their audio characteristics using
a content-based machine learning model.

Follow these instructions for best results:

1. Enter a song name in the selection box.  
   You do not need to scroll through the full list.  
   Instead, click the box and start typing the song name.

2. Capitalize the first letter of each word in the song title.  
   This ensures a clean match with the dataset.  
   Example: "In The End" rather than "in the end".

3. After selecting the song, choose the correct artist from the artist box.  
   Some songs appear multiple times in Spotify playlists, so matching the correct
   artist improves recommendation accuracy.

4. Press the Recommend button to view the top ten recommendations.  
   Each suggested song includes links to YouTube and Spotify searches.

5. The model restricts recommendations to the same genre group to ensure that
   the output remains musically coherent.

If you encounter a case where no recommendations appear, please check the
spelling of the track name and ensure that both the song and artist exist in the dataset.

Also, please be patient. You may have to wait for the URL/selection box to render, especially when you update your prompt. 
""")
