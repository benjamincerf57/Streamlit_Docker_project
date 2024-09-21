import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Benjamin's project! 👋")

st.markdown("""
Welcome to my first Streamlit project!

This project was conducted as part of the 'Tools for Data Scientist' course taught at HEC Paris, and aims to apply the knowledge gained about Docker and Streamlit.

The Data Science project used is one of my first projects, which simply involves predicting the sklearn 'digits' dataset. The method and model chosen were inspired by Gaël Varoquaux, director of sklearn.

Here is the link to my personal GitHub: [My GitHub](https://github.com/benjamincerf57)
""")