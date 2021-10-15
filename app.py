import streamlit as st


championship_csv = {'ligue-1': 'ligue-1_data_2002_2019',
                   'ligue-2': 'ligue-2_data_2002_2019',
                   'serie-A': 'serie-a_data_2004_2019',
                   'bundesliga': 'bundesliga_data_2004_2019',
                   'premier-league': 'premier-league_data_2004_2019',
                   'liga':'liga_data_2004_2019'}

st.title('Soccer : what is the final ranking ?')

@st.cache
def load_data(championship: str):
    """extract data"""
    pass

@st.cache
def preprocess(data_df, model_type='naive'):
    """Preprocess data according to the choice of the model"""
    pass

# EDA PART
st.subheader("Exploratory Data Analysis")
# choose the championship

# show basic eda

# show eda plots

# PREDICT PART
st.subheader("Predict the ranking")

# Naive model

# choose the model type : regression, classification or ranking algorithm

# provide the input 

# show prediction