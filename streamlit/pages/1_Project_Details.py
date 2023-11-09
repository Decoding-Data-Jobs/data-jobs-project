import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import time
import numpy as np

st.set_page_config(page_title="Project Details", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

@st.cache_resource
def load_full_data():
    return pd.read_csv('../support_files/prepped_jobs.csv')

jobs_df_cleaned = load_full_data()

def plot_monthly_postings():
    # Load the data
    jobs_df_cleaned = pd.read_csv('../support_files/prepped_jobs.csv')

    # Make a new df with date time index
    jobs_df_cleaned_date = jobs_df_cleaned.copy()

    # Make posting_created date time
    jobs_df_cleaned_date['posting_created'] = pd.to_datetime(jobs_df_cleaned_date['posting_created'])

    # Set index to posting_created
    jobs_df_cleaned_date.set_index('posting_created', inplace=True)

    # Resample 'posting_created' to monthly frequency and count the number of postings
    monthly_postings = jobs_df_cleaned_date.resample('M').size()

    # Exclude November 2023
    monthly_postings = monthly_postings.loc[~((monthly_postings.index.month == 11) & (monthly_postings.index.year == 2023))]

    # Initialize the chart with the first data point
    chart = st.line_chart(monthly_postings[:1])

    # Display a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Add the rest of the data points in a loop
    for i in range(1, len(monthly_postings)):
        # Update the progress bar and status text
        progress_bar.progress(i + 1)
        status_text.text(f"{(i + 1) / len(monthly_postings):.2%} Complete")

        # Add the next data point to the chart
        chart.add_rows(monthly_postings[i:i+1])

        # Pause for a moment
        time.sleep(1)

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()


def about_page():
    # Page Title
    st.title("About This Project")

    st.markdown("""
    This project is a data visualization tool for analyzing job skills data. 
    It allows users to select a category of skills and the number of top skills to display, 
    and it presents the data in a bar chart. The color of the bars represents the average yearly salary 
    associated with each skill.
    """)

    st.markdown("""
    ### Preparation

    Words
    """)

    st.dataframe(jobs_df_cleaned)

    st.markdown("""

    ### Exploration

    Words

    ### Modeling

    Words

    ### Delivery

    Words

    """)

    plot_monthly_postings()

    st.markdown("""
    ---
    """)

# Call the function to display the about page
about_page()