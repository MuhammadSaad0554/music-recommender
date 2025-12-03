import streamlit as st

st.title("Feedback")

st.write("""
Thank you for trying the music recommendation system.
Your feedback will help improve the recommendation algorithm
and overall user experience.
""")

st.write("Please submit your feedback using the link below:")

st.markdown(
    "[Submit Feedback Survey](https://docs.google.com/forms/d/e/1FAIpQLSffGug9JtByMTMvWocdeVIOfhY4Fhy34vKl2ndHwQ2xgUkm0g/viewform?usp=sharing)"
)

st.write("""
After submitting the form, you can return to the Recommender tab to test more songs.
""")
