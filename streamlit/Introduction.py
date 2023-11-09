
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Introduction", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

# TO RUN THIS:
# IN TERMINAL PUT:
# streamlit run Introduction.py

# DEMO APP:
# streamlit hello

# Load the data
@st.cache_resource
def load_data():
    return pd.read_csv('../support_files/working_docs/top_skills.csv')

top_skills_df = load_data()

@st.cache_resource
def load_full_data():
    return pd.read_csv('../support_files/working_docs/jobs_prepped.csv')

jobs_df_cleaned = load_full_data()


st.markdown("""
<div style="text-align: center"> 

# Decoding Data Jobs

(picture here)

</div>
""", unsafe_allow_html=True)

st.markdown("""
# Introduction

We are students at CodeUp that are currently looking for Data Analyst Employment. We created this dashboard to help us understand the Data Analyst job market in the United States. We hope that this dashboard will help us and others like us to better understand what skills are needed to be successful in this field.

# Project Summary

More words and things

### Planning

Words

""")

@st.cache_resource
def load_original_data():
    return pd.read_csv('../support_files/working_docs/jobs.csv')

data = load_original_data()
csv1 = data.to_csv(index=False)

st.download_button(
    label="Download Original Data (156MB)",
    data=csv1,
    file_name="data_jobs.csv",
    mime="csv")

st.dataframe(data)