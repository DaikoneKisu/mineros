import streamlit as st

pg = st.navigation([st.Page("models.py"), st.Page("eda.py")])
pg.run()
