
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
@st.cache_data
def load_data():
    return pd.read_csv('../support_files/top_skills.csv')

top_skills_df = load_data()

@st.cache_data
def load_full_data():
    return pd.read_csv('../support_files/prepped_jobs.csv')

@st.cache_data
def load_job_data():
    return pd.read_csv('../support_files/jobs_stripped.csv')

jobs = load_job_data()

st.image('pages/images/wordcloud.png')

st.markdown("""
<h2 style="font-size:30px; text-align:center;">We developed this dynamic tool that empowers aspiring data analysts to navigate the job market with precision and confidence.</h2>

<h2 style="font-size:20px; text-align:center;">Our project uses 33,000 job postings that were scraped from Google and then processed with machine learning models to validate the results. We're not just sifting through text; we're decoding the jargon of job descriptions to help future data analysts on the hunt for their dream roles. We provide them with a competitive edge by revealing the skills, qualifications, and trends that matter most.</h2>
""", unsafe_allow_html=True)

st.markdown('Original DataSet We Worked With:')
# Read in and display jobs_stripped.csv
st.write(jobs)