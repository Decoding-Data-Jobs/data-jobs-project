import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Popular Skills", page_icon="💰", layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('https://drive.google.com/uc?export=download&id=1Ai3JtDGpIKdRkt01WHAHHH63uJ2ITy4n')

@st.cache_data
def load_full_data():
    return pd.read_csv('https://drive.google.com/uc?export=download&id=1S0GQlhjUc3WN0nLWIVeiSRDvWWTDUdE1')

jobs_df_cleaned = load_full_data()

top_skills_df = load_data()

# Picked out keywords based on all keywords (only looked words with 100+ occurrences)
keywords_programming = [
'sql', 'python', 'r', 'c', 'c#', 'javascript', 'js',  'java', 'scala', 'sas', 'matlab', 
'c++', 'c/c++', 'perl', 'go', 'typescript', 'bash', 'html', 'css', 'php', 'powershell', 'rust', 
'kotlin', 'ruby',  'dart', 'assembly', 'swift', 'vba', 'lua', 'groovy', 'delphi', 'objective-c', 
'haskell', 'elixir', 'julia', 'clojure', 'solidity', 'lisp', 'f#', 'fortran', 'erlang', 'apl', 
'cobol', 'ocaml', 'crystal', 'javascript/typescript', 'golang', 'nosql', 'mongodb', 't-sql', 'no-sql',
'visual_basic', 'pascal', 'mongo', 'pl/sql',  'sass', 'vb.net', 'mssql',
]
# Pick out ML Algorithm keywords
keywords_ML_Algorithms = [x.lower() for x in ['regression','clustering', 'classification', 'predictive', 'prediction','decision trees',
       'Decision Trees, Random Forests',
       'Convolutional Neural Networks','CNN',
       'Gradient Boosting Machines (xgboost, lightgbm, etc)',
       'Bayesian Approaches', 'Dense Neural Networks (MLPs, etc)','DNN',
       'Recurrent Neural Networks','RNN',
       'Transformer Networks (BERT, gpt-3, etc)', 'Graph Neural Networks','Transformer'
       'Autoencoder Networks (DAE, VAE, etc)',
       'Generative Adversarial Networks', 'None',
       'Evolutionary Approaches',] ]
# Viz keywords
keyword_viz = [x.lower() for x in ['Matplotlib', 'Seaborn', 'Plotly',
       'Ggplot', 'None', 'Shiny', 'Geoplotlib', 'Bokeh',
       'D3 js', 'Other', 'Leaflet / Folium', 'Pygal', 'Altair',
       'Dygraphs', 'Highcharter', 'tableau',  'Microsoft Power BI', 'Google Data Studio',
       'Amazon QuickSight', 'Qlik Sense', 'Other',
       'Microsoft Azure Synapse ', 'Looker', 'Alteryx ',
       'SAP Analytics Cloud ', 'TIBCO Spotfire', 'Domo', 'Sisense ',
       'Thoughtspot '] ]
# Computer vision and nlp
keyword_cvnlp = ['computer vision','natural language processing']
# Big data keywords
keyword_big_data = ['mysql', 'postgresql', 'microsoft sql', 'sqlite', 'mongodb',
                    'bigquery', 'oracle database', 'azure sql', 'amazon rds', 'google cloud sql', 'snowflake']
# More big data
keyword_big_data_2 = [x.lower() for x in  ['MySQL ', 'PostgreSQL ', 'Microsoft SQL Server ', 'SQLite ',
       'MongoDB ', 'None', 'Google Cloud BigQuery ', 'Oracle Database ',
       'Microsoft Azure SQL Database ', 'Amazon RDS ',
       'Google Cloud SQL ', 'Snowflake ', 'Amazon Redshift ',
       'Amazon DynamoDB ', 'Other', 'IBM Db2 '] ]
# Business Intelligence keywords
keyword_bi = [x.lower() for x in ['tableau',  'Power BI', 'Power_bi', 'Google Data Studio',
       'QuickSight', 'Qlik Sense', 'Other',
       'Azure Synapse ', 'Looker', 'Alteryx ',
       'SAP Analytics Cloud ', 'TIBCO Spotfire', 'Domo', 'Sisense ',
       'Thoughtspot '] ]
# More business intelligence
keyword_bi_2 = [x.lower() for x in ['tableau',  'Microsoft Power BI', 'Google Data Studio',
       'Amazon QuickSight', 'Qlik Sense', 'Other',
       'Microsoft Azure Synapse ', 'Looker', 'Alteryx ',
       'SAP Analytics Cloud ', 'TIBCO Spotfire', 'Domo', 'Sisense ',
       'Thoughtspot '] ]
# Analyst tools
keywords_analyst_tools = [
'excel', 'tableau',  'word', 'powerpoint', 'looker', 'powerbi', 'outlook', 'azure', 'jira', 'twilio',  'snowflake', 
'shell', 'linux', 'sas', 'sharepoint', 'mysql', 'visio', 'git', 'mssql', 'powerpoints', 'postgresql', 'spreadsheets',
'seaborn', 'pandas', 'gdpr', 'spreadsheet', 'alteryx', 'github', 'postgres', 'ssis', 'numpy', 'power_bi', 'spss', 'ssrs', 
'microstrategy',  'cognos', 'dax', 'matplotlib', 'dplyr', 'tidyr', 'ggplot2', 'plotly', 'esquisse', 'rshiny', 'mlr',
'docker', 'linux', 'jira',  'hadoop', 'airflow', 'redis', 'graphql', 'sap', 'tensorflow', 'node', 'asp.net', 'unix',
'jquery', 'pyspark', 'pytorch', 'gitlab', 'selenium', 'splunk', 'bitbucket', 'qlik', 'terminal', 'atlassian', 'unix/linux',
'linux/unix', 'ubuntu', 'nuix', 'datarobot',
]
# Cloud tools
keywords_cloud_tools = [
'aws', 'azure', 'gcp', 'snowflake', 'redshift', 'bigquery', 'aurora','amazon','ec2','s3',
]
# Not using
keywords_general_tools = [
'microsoft', 'slack', 'apache', 'ibm', 'html5', 'datadog', 'bloomberg',  'ajax', 'persicope', 'oracle', 
]
# Not using
keywords_general = [
'coding', 'server', 'database', 'cloud', 'warehousing', 'scrum', 'devops', 'programming', 'saas', 'ci/cd', 'cicd', 
'ml', 'data_lake', 'frontend',' front-end', 'back-end', 'backend', 'json', 'xml', 'ios', 'kanban', 'nlp',
'iot', 'codebase', 'agile/scrum', 'agile', 'ai/ml', 'ai', 'paas', 'machine_learning', 'macros', 'iaas',
'fullstack', 'dataops', 'scrum/agile', 'ssas', 'mlops', 'debug', 'etl', 'a/b', 'slack', 'erp', 'oop', 
'object-oriented', 'etl/elt', 'elt', 'dashboarding', 'big-data', 'twilio', 'ui/ux', 'ux/ui', 'vlookup', 
'crossover',  'data_lake', 'data_lakes', 'bi', 
]

keywords = keywords_programming + keywords_ML_Algorithms + keywords_analyst_tools + keywords_cloud_tools 

def plot_skills_data(top_skills_df, keywords, keywords_programming, keywords_ML_Algorithms, keyword_viz, keyword_bi, keyword_bi_2, keywords_cloud_tools, keyword_big_data):
    # Slider button group for top N skills
    top_n = st.slider(
    'Select Number of Top Skills:',
    min_value=10, max_value=100, value=20, step=10)

    
    # Dropdown widget for skill category
    skill_list = st.selectbox(
        'Select Skill Category:',
        [('All Skills', keywords), ('Programming Languages', keywords_programming), 
         ('ML Algorithms', keywords_ML_Algorithms), ('Visualization Tools', keyword_viz + keyword_bi + keyword_bi_2), 
         ('Big Data & Cloud', keywords_cloud_tools + keyword_big_data)],
        format_func=lambda x: x[0]  # Display only the skill category in the dropdown
    )

    # Filter DataFrame based on selected skills
    df = top_skills_df[top_skills_df['skill'].isin(skill_list[1])]

    # Select top N skills based on frequency
    df = df.nlargest(top_n, 'frequency (%)')

    with st.container():

        # Display Title
        st.markdown(f"<h1 style='font-size:40px; text-align: center; color: white;'>{skill_list[0]}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='font-size:30px; text-align: center; color: white;'><i>Salaries and Popularity</i></h2>", unsafe_allow_html=True)

        fig = px.bar(df, x='skill', y='frequency (%)',
                    color='avg_yearly_salary', color_continuous_scale='Blues')

        fig.update_traces(textfont_size=40,
                        hovertemplate='''<b>Skill:</b> %{x}<br><b>Frequency:</b> %{y}%<br><b>Average Salary:</b> $%{marker.color}<extra></extra>''',
                        hoverlabel=dict(font_size=30, font_color='#FF4B4B'))

        fig.update_layout( 
            font_color="white",
            coloraxis_colorbar=dict(title="Average Annual Salary", title_font=dict(size=30), tickfont=dict(size=20)),
            height=600,
            width=1200,
            yaxis=dict(title='Frequency (%)', title_font=dict(size=30), tickfont=dict(size=20)),
            xaxis=dict(title='Skill', title_font=dict(size=30), tickfont=dict(size=20)),       
            )


        st.plotly_chart(fig, use_container_width=True)

# Call the function with your data
plot_skills_data(top_skills_df, keywords, keywords_programming, keywords_ML_Algorithms, keyword_viz, keyword_bi, keyword_bi_2, keywords_cloud_tools, keyword_big_data)