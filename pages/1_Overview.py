
import streamlit as st

st.title("🎵 Music Recommendation System")
st.subheader("By Muhammad Saad")

st.markdown("""
## 📘 Project Overview

This project builds a content-based music recommender using:

- StandardScaler  
- PCA  
- K-Means (k = 30)  
- Cosine Similarity  
- Hybrid filtering (genre + cluster)

---

## 📝 How to Use
- Go to the **Recommender** page
- Select a song
- Select the correct artist
- Click *Recommend*

You will receive 10 similar songs with Spotify and YouTube links.
""")
