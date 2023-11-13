import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Average Skill Salaries", page_icon="💰", layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

def interactive_skill_salary():
    
    
    # Load the data
    top_skills_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1Ai3JtDGpIKdRkt01WHAHHH63uJ2ITy4n')
    jobs_df_cleaned = pd.read_csv('https://drive.google.com/uc?export=download&id=1S0GQlhjUc3WN0nLWIVeiSRDvWWTDUdE1')

    # Get the top skills sorted alphabetically
    sorted_skills = sorted(top_skills_df['skill'].unique())

    # Interactive dropdown widget
    skill = st.selectbox('Skill:', sorted_skills)

    if skill:  # Don't plot if skill is None
        # Filter the dataframe for jobs that mention the selected skill and have a salary
        skill_df = jobs_df_cleaned[jobs_df_cleaned['description_tokens'].apply(lambda x: skill in x) & jobs_df_cleaned['avg_salary'].notna()].sort_values(by='avg_salary')
        # Filter the dataframe for jobs that mention the selected skill but do not have a salary
        skill_df_no_salary = jobs_df_cleaned[jobs_df_cleaned['description_tokens'].apply(lambda x: skill in x) & jobs_df_cleaned['avg_salary'].isna()].sort_values(by='avg_salary')
        # Reset the index
        skill_df = skill_df.reset_index(drop=True)

        # Calculate the average salary for the selected skill
        avg_salary = skill_df['avg_salary'].mean()

        # Display in a container
        with st.container():

            # Display title
            st.markdown(f"<h1 style='font-size:40px; text-align: center; color: white;'>Salary Distribution of Individual Skills</h1>", unsafe_allow_html=True)
            
            # Create a bar plot
            fig = px.bar(skill_df, x=skill_df.index, y='avg_salary', 
                        hover_data=['avg_salary', 'description_tokens', 'company_name'],
                        labels={'avg_salary':'Salary ($)', 'index':'Posts WITH Salary'},
                        color='avg_salary',
                        color_continuous_scale='Blues', title=f'{str.capitalize(skill)}')

            fig.update_traces(
                textfont_size=40,
                hovertemplate='''<b>Company Name:</b><br>%{customdata[1]}<br><br><b>Description Tokens:</b><br>%{customdata[0]}<br><br><b>Average Salary:</b><br>$%{marker.color}<extra></extra>''',
                hoverlabel=dict(font_size=20, font_color='#FF4B4B')
            )
            # Add a line for the average salary
            fig.add_shape(
                type='line',
                line=dict(dash='dash', color='white'),
                y0=avg_salary,
                y1=avg_salary,
                x0=0,
                x1=1,
                xref='paper',
                yref='y',
            )

            # Add a text label for the average salary
            fig.add_annotation(
                y=avg_salary + 15000,
                x=0,
                xref='paper',
                yref='y',
                text=f'Average Salary: ${avg_salary:.2f}',
                showarrow=False,
                font=dict(size= 20, color='white')
            )

            # Update layout
            fig.update_layout(
                coloraxis_colorbar=dict(title='Salary', title_font=dict(size=20), tickfont=dict(size=18)),
                autosize=True,
                hovermode='closest',
                showlegend=False,
                yaxis=dict(title='Salary ($)', title_font=dict(size=30), tickfont=dict(size=20)),
                xaxis=dict(title='Jobs Postings with Salary<br><i>(Hover for Details)</i>', title_font=dict(size=30), tickfont=dict(size=20)),
                height=600,
                title=dict(x=0.5, y=0.9, xanchor='center', yanchor='top', font=dict(size=50))
            )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f'Posts With Salary: {len(skill_df)}')
            st.subheader(f'Posts Without Salary: {len(skill_df_no_salary)}')
        
# Call the function
interactive_skill_salary()