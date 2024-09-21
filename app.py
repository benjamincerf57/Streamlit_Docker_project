import streamlit as st
import pandas as pd
import numpy as np

st.title('Mon Projet Data Science')

data = pd.DataFrame(
    np.random.randn(100, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(data)
