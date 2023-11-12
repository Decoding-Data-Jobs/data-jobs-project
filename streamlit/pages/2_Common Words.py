import streamlit as st
import pandas as pd
import plotly.express as px

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('../support_files/word_percentages.csv')
    return data

data = load_data()

# Selecting the number of skills
num_skills = st.slider('Select number of top skills', min_value=10, max_value=50, value=20, step=10)

# Selecting the role
role = st.selectbox('Select role', ['Overall', 'Data Analyst', 'Data Scientist', 'Data Engineer'])

# Map each role to the other two roles
role_map = {
    'Data Analyst': ['Data Scientist', 'Data Engineer'],
    'Data Scientist': ['Data Analyst', 'Data Engineer'],
    'Data Engineer': ['Data Analyst', 'Data Scientist'],
    'Overall': []  # Add an empty list for 'Overall' case
}

# Show me columns where selected role is 10 greater than the other two roles
if role != 'Overall':  # Check if the role is not 'Overall'
    other_roles = role_map[role]
    filtered_data = data[(data[role] > data[other_roles[0]] + 10) & (data[role] > data[other_roles[1]] + 10)]
else:
    filtered_data = data

# Filter the data based on the selection
top_skills = filtered_data.nlargest(num_skills, role)

# Plotting the data using Plotly
fig = px.bar(top_skills, x='Word', y=role)

st.markdown(f'<h1 style="font-size:40px; text-align:center;">{num_skills} Unique Words for {role} Descriptions</h1>', unsafe_allow_html=True)

if role != 'Overall':
    st.markdown(f'<h2 style="font-size:20px; text-align:center;"><i>At least 10% more than other titles</i></h2>', unsafe_allow_html=True)

# Update hovertemplate based on the role
if role == 'Overall':
    hovertemplate = '<b>Word:</b> %{x} <br><b>Frequency:</b> %{y}% <br><extra></extra>'
else:
    other_roles = role_map[role]
    hovertemplate = '<b>Word:</b> %{x} <br><br><b>' + role + ' Frequency:</b> %{y}%<br>' + \
                    '<b>' + other_roles[0] + ' Frequency:</b> %{customdata[0]}%<br>' + \
                    '<b>' + other_roles[1] + ' Frequency:</b> %{customdata[1]}%<br><extra></extra>'

    fig.update_traces(customdata=top_skills[[other_roles[0], other_roles[1]]])


fig.update_traces(textfont_size=40,
                  hovertemplate=hovertemplate,
                  hoverlabel=dict(font_size=30, font_color="#FF4B4B"))

fig.update_layout(
    yaxis=dict(title='Frequency (%)', title_font=dict(size=30), tickfont=dict(size=20)),
    xaxis=dict(title='Skill', title_font=dict(size=30), tickfont=dict(size=20)),
    height=600
)

st.plotly_chart(fig, use_container_width=True)