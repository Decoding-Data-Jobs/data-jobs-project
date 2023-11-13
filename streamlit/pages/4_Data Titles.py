import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import ast


st.set_page_config(page_title="Data Titles", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

@st.cache_data
def load_full_data():
    df = pd.read_csv('https://drive.google.com/uc?export=download&id=1S0GQlhjUc3WN0nLWIVeiSRDvWWTDUdE1')
    return df

jobs_df_cleaned = load_full_data()

jobs_df_cleaned['description_tokens'] = jobs_df_cleaned['description_tokens'].apply(ast.literal_eval)

def get_top_skills(df, qty, title_cleaned=None):
    # If a job title is provided, filter the DataFrame
    if title_cleaned:
        df = df[df['title_cleaned'] == title_cleaned]
    
    # Initialize an empty dictionary to store the skills and their counts
    skills_counts = {}

    # Loop over the values in the 'description_tokens' column
    for val in df.description_tokens.values:
        # Check if 'description_tokens' is not an empty list
        if val:
            # Convert the list of skills in each posting to a set to remove duplicates
            unique_skills = set(val)
            # Increment the count for each unique skill
            for skill in unique_skills:
                if skill in skills_counts:
                    skills_counts[skill] += 1
                else:
                    skills_counts[skill] = 1
    
    # Get the top skills and their counts
    top_skill_count = sorted(skills_counts.items(), key=lambda x: -x[1])[:qty]
    
    # Separate the skills and counts into two lists
    top_skills = list(map(lambda x: x[0], top_skill_count))
    top_counts = list(map(lambda x: x[1], top_skill_count))
    
    # Initialize an empty list to store the average salaries
    salaries = []
    
    # Loop over the top skills and calculate their average salary
    for skill in top_skills: 
        skill_df = df[df.description_tokens.apply(lambda x: skill in x)]
        if skill_df.avg_salary.isna().all():  # If all salaries for this skill are NaN
            salaries.append(np.nan)  # Append NaN to the salaries list
        else:
            salaries.append(skill_df.avg_salary.mean())
    
    # Create a DataFrame with the top skills, their number of postings, and their average yearly salary
    top_skills_df = pd.DataFrame({
        "skill": top_skills, 
        "number_of_postings": top_counts,
        "avg_yearly_salary": [round(s) if s == s else np.nan for s in salaries]  # Only round the salary if it is not NaN
    })
    
    # Calculate the frequency of each skill
    top_skills_df['frequency (%)'] = round((top_skills_df['number_of_postings'] / df.shape[0]) * 100, 2)
    
    # Sort the DataFrame by average yearly salary in descending order
    top_skills_df = top_skills_df.sort_values("number_of_postings", ascending=False)

    # Remove 'none'
    top_skills_df = top_skills_df[top_skills_df['skill'] != 'none']

    # Round values to 2 decimals
    top_skills_df['avg_yearly_salary'] = top_skills_df['avg_yearly_salary'].round()
    
    return top_skills_df

def plot_top_skills(df, qty, title_cleaned=None):
    # Get the top skills
    top_skills_df = get_top_skills(df, qty, title_cleaned)

    # Sort by frequency
    top_skills_df.sort_values(by='frequency (%)', ascending=False, inplace=True)

    # Plot it
    fig = px.bar(top_skills_df, x='skill', y='frequency (%)', color='avg_yearly_salary', color_continuous_scale='Blues')

    fig.update_traces(textfont_size=40, hovertemplate='''<b>Skill:</b> %{x}<br><b>Frequency:</b> %{y}%<br><b>Average Annual Salary:</b> $%{marker.color}<extra></extra>''', hoverlabel=dict(font_size=30, font_color="#FF4B4B"))

    fig.update_layout(
        yaxis=dict(title='Frequency (%)',title_font=dict(size=30), tickfont=dict(size=20)),
        xaxis=dict(title='Skill', title_font=dict(size=30), tickfont=dict(size=20)),
        coloraxis_colorbar=dict(title="Average Annual Salary", title_font=dict(size=20), tickfont=dict(size=18)),
        height=600
    )
    return fig

# Interactive elements
qty = st.slider('Select number of top skills', 10, 50, 10, 10)
titles = jobs_df_cleaned['title_cleaned'].unique().tolist()
titles = [title for title in titles if title != 'Other']
title_cleaned = st.selectbox('Select job title', ['All Jobs'] + titles)

if title_cleaned == 'All Jobs':
    title_cleaned = None
with st.container():
    # Display Title
    st.markdown(f"<h1 style='font-size:40px; text-align: center; color: white;'>{title_cleaned if title_cleaned else 'All Jobs'} Skills</h1>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='font-size:30px; text-align: center; color: white;'><i>Salaries and Popularity</i></h2>", unsafe_allow_html=True)
fig = plot_top_skills(jobs_df_cleaned, qty, title_cleaned)
st.plotly_chart(fig, use_container_width=True)