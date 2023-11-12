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
num_skills = st.slider('Select number of top skills', min_value=5, max_value=100, value=10, step=1)

# Selecting the role
role = st.selectbox('Select role', ['Overall', 'Data Analyst', 'Data Scientist', 'Data Engineer'])

# Map each role to the other two roles
role_map = {
    'Data Analyst': ['Data Scientist', 'Data Engineer'],
    'Data Scientist': ['Data Analyst', 'Data Engineer'],
    'Data Engineer': ['Data Analyst', 'Data Scientist']
}

# Show me columns where selected role is 20 greater than the other two roles
if role in role_map:
    other_roles = role_map[role]
    st.write(data[(data[role] > data[other_roles[0]] + 10) & (data[role] > data[other_roles[1]] + 10)].head(10))
else:
    role = 'Overall'

# Filter the data based on the selection
top_skills = data.nlargest(num_skills, role)

# Plotting the data using Plotly
fig = px.bar(top_skills, x='Word', y=role, title=f'Top {num_skills} Skills for {role.capitalize()}')

fig.update_traces(textfont_size=40,
                hovertemplate='''
<b>Word:</b> %{x} <br> \
<b>Frequency:</b> %{y}% <br><extra></extra>''',
                hoverlabel=dict(font_size=20))

st.plotly_chart(fig, use_container_width=True)