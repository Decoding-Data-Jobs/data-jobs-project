import streamlit as st


st.set_page_config(page_title="Personal Page", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items={
    "About": "https://www.linkedin.com/in/zschmitz https://github.com/Zacharia-Schmitz"
})

def personals_page():
    st.image("thanks.png")
    st.markdown("""

    <h2 style="font-size:20px; text-align:center;"><i>Contact us below for any inquiries.</i></h2>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2, col3, col4, col5 = st.columns(5)

    col2.image("zac.png", use_column_width=True)
    col4.image("click.png", use_column_width=True)

    col2.markdown("<h3 style='text-align: center; color: white;'>Zacharia Schmitz</h3>", unsafe_allow_html=True)
    col4.markdown("<h3 style='text-align: center; color: white;'>Joshua Click</h3>", unsafe_allow_html=True)

    col2.markdown("""
    <div style='text-align:center'>
    <a href='mailto:schmitz.zacharia@gmail.com'>
    <img src='https://img.shields.io/badge/Gmail-%23EA4335.svg?style=plastic&amp;logo=gmail&amp;logoColor=white' alt='Gmail' style='max-width: 100%;'></a> 
    <a href='https://www.linkedin.com/in/zschmitz/' rel='nofollow'>
    <img src='https://img.shields.io/badge/LinkedIn-%230A66C2.svg?style=plastic&amp;logo=linkedin&amp;logoColor=white' alt='LinkedIn' style='max-width: 100%;'></a> 
    <a href='https://github.com/Zacharia-Schmitz'>
    <img src='https://img.shields.io/badge/GitHub-%23181717.svg?style=plastic&amp;logo=github&amp;logoColor=white' alt='Github' style='max-width: 100%;'></a>
    </div><br>
    """, unsafe_allow_html=True)

    col4.markdown("""
    <div style='text-align:center'>
    <a href='mailto:joshua.click25@gmail.com'>
    <img src='https://img.shields.io/badge/Gmail-%23EA4335.svg?style=plastic&amp;logo=gmail&amp;logoColor=white' alt='Gmail' style='max-width: 100%;'></a> 
    <a href='https://www.linkedin.com/in/joshua-r-click/' rel='nofollow'>
    <img src='https://img.shields.io/badge/LinkedIn-%230A66C2.svg?style=plastic&amp;logo=linkedin&amp;logoColor=white' alt='LinkedIn' style='max-width: 100%;'></a> 
    <a href='https://github.com/Joshua-Click'>
    <img src='https://img.shields.io/badge/GitHub-%23181717.svg?style=plastic&amp;logo=github&amp;logoColor=white' alt='Github' style='max-width: 100%;'></a>
    </div><br>
    """, unsafe_allow_html=True)



# Call the function to display the about page
personals_page()