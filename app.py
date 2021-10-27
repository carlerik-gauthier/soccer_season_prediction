from os import name
import streamlit as st
# https://docs.streamlit.io/library/get-started/main-concepts
# https://docs.streamlit.io/library/get-started/create-an-app 
from utils import is_available, train_model, retrieve_model

# TODO : 1. connection to model// 2. EDA part

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
st.header("Exploratory Data Analysis. Data comes from l'Equipe website and runs from season 2004-2005 to 2018-2019")
see_eda = st.selectbox(label="Do you want to see some EDA ?", options=['yes', 'no'])
# choose the championship
if see_eda == 'yes':
    championship_choices_list = st.multiselect(label="Select the championship you want to see",
    options=championship_csv.keys())
    # show basic eda

    # show eda plots
    option_1 = st.selectbox("Do you want to see ... ?", options=['yes', 'no'])
    option_2 = st.selectbox("Do you want to see ... ?", options=['yes', 'no'])
    option_3 = st.selectbox("Do you want to see ... ?", options=['yes', 'no'])
    option_4 = st.selectbox("Do you want to see ... ?", options=['yes', 'no'])
    option_5 = st.selectbox("Do you want to see ... ?", options=['yes', 'no'])

# PREDICT PART
st.header("Predict the ranking")
st.subheader("Naive model is included")
# Naive model

# choose the model type : regression, classification or ranking algorithm
model_type_option = st.selectbox(label="Please choose the type of algorithm used by the ranker. More than one can be selected", 
options=['regression', 'classification', 'ranking'])
# Get model :
use_pretrained = False
model_name = "{model_type}_ranker".format(model_type=model_type_option)
#  --- check if pretrained model is available 
model_available = is_available(module='saved_models', name=model_name)

if model_available: 
    choice = st.selectbox(label="Do you want to use an already trained model ?", options=['yes', 'no'])
    use_pretrained = choice=='yes'
# -- retrieve a pretrained model if requested and available else train the model
if use_pretrained:
    model = retrieve_model(module="saved_models", name=model_name)
    pass
else:
    # train model
    model = train_model(model_type=model_type_option)
    pass

# provide the input
# -- show the head of expected input dataframe 
st.dataframe(data=..., height=5)
st.download_button(label="Provide your input : a csv file with the above format collecting all games played in one season until a specified leg")
# input_path = st.text_input(label="Enter the path to you csv file.",
# value="")
# show prediction

