"""
Making an app based upon the streamlit package. Documentation can be found there :
https://docs.streamlit.io/library/get-started/main-concepts
https://docs.streamlit.io/library/api-reference
https://docs.streamlit.io/library/get-started/create-an-app
"""
import os

import pandas as pd
import streamlit as st
from copy import deepcopy
from utils import is_available, train_model, retrieve_model
from preprocess.soccer_data import prepare_data
from preprocess.predictor_preprocess import build_data, get_pivoted
# TODO : 1. connection to model// 2. EDA part

season_options = ['{start_year}-{end_year}'.format(start_year=year, end_year=year+1) for year in range(2004, 2019)]

championship_csv = {'ligue-1': 'ligue-1_data_2002_2019',
                    'ligue-2': 'ligue-2_data_2002_2019',
                    'serie-A': 'serie-a_data_2004_2019',
                    'bundesliga': 'bundesliga_data_2004_2019',
                    'premier-league': 'premier-league_data_2004_2019',
                    'liga': 'liga_data_2004_2019'}

st.title('Soccer : what is the final ranking ?')


@st.cache
def load_data(league: str):
    """extract data"""
    csv_file = championship_csv.get(league, 'ligue-1_data_2002_2019')
    csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    return prepare_data(csv_path=csv_path, raw=True)


@st.cache
def preprocess(data_df: pd.DataFrame, model_type: str = 'naive', breaking_leg: int = 27):
    """Preprocess data according to the choice of the model"""
    if model_type == 'classification':
        return get_pivoted(data=data_df, break_leg=breaking_leg)
    elif model_type == 'regression':
        return build_data(historical_data=data_df, break_leg=breaking_leg)
    elif model_type == 'ranking':
        return get_pivoted(data=data_df, break_leg=breaking_leg)
    else:
        # will process Naive model
        return build_data(historical_data=data_df, break_leg=breaking_leg)


st.markdown("## Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")
placeholder = st.empty()
placeholder_1 = st.empty()
placeholder_2 = st.empty()
# EDA PART
st.sidebar.markdown("#### EDA")
see_eda = st.sidebar.selectbox(label="Do you want to see some EDA ?", options=['yes', 'no'], index=1)
# choose the championship
if see_eda == 'yes':
    placeholder.markdown("#### Exploratory Data Analysis")
    championship_choices_list = st.sidebar.multiselect(label="Select the championship you want to see",
                                                       options=championship_csv.keys())
    # get data

    championship_data = {champ: load_data(league=champ) for champ in championship_choices_list}
    # show basic eda
    placeholder_1.write("Basic EDA")
    # show eda plots
    option_1 = st.sidebar.selectbox(label="     Do you want to see 1st plot ?", options=['yes', 'no'], index=1)
    option_2 = st.sidebar.selectbox(label="     Do you want to see 2nd plot ?", options=['yes', 'no'], index=1)
    option_3 = st.sidebar.selectbox(label="     Do you want to see 3rd plot ?", options=['yes', 'no'], index=1)
    option_4 = st.sidebar.selectbox(label="     Do you want to see 4th plot ?", options=['yes', 'no'], index=1)
    option_5 = st.sidebar.selectbox(label="     Do you want to see 5th plot ?", options=['yes', 'no'], index=1)

# PREDICT PART
st.sidebar.empty()
st.sidebar.markdown("### Prediction")
start_prediction = st.sidebar.selectbox(label="Are you ready to predict the next champion ?",
                                        options=["yes", "no"],
                                        index=1)
if start_prediction == 'yes':
    placeholder_1.empty()
    placeholder_2.empty()
    placeholder.markdown("##### Predict the ranking")

    st.sidebar.markdown("#### Naive model is included")
    # choose championship
    championship = st.sidebar.selectbox(label="Choose your championship you want to predict on",
                                        options=championship_csv.keys())
    # choose the model type : regression, classification or ranking algorithm
    model_type_option = st.sidebar.selectbox(
        label="Please choose the type of algorithm used by the ranker. More than one can be selected",
        options=['regression', 'classification', 'ranking']
        )

    # get break leg
    break_leg = st.sidebar.slider(label="How many legs have been played so far ?",
                                  min_value=1,
                                  max_value=34 if championship == 'bundesliga' else 38,
                                  step=1,
                                  value=27)
    # Get model :
    use_pretrained = False
    model_name = "{model_type}_{championship}_leg{break_leg}_ranker".format(model_type=model_type_option,
                                                                            championship=championship,
                                                                            break_leg=break_leg)
    #  --- check if pretrained model is available
    model_available = is_available(module_path='saved_models', file_name=model_name)

    if model_available:
        choice = st.sidebar.selectbox(label="Do you want to use an already trained model ?", options=['yes', 'no'])
        use_pretrained = choice == 'yes'
    # -- retrieve a pretrained model if requested and available else train the model
    if use_pretrained:
        model = retrieve_model(module_path="saved_models", file_name=model_name)
    else:
        # train model
        training_seasons = st.sidebar.multiselect(
            label="Select the seasons on which the model should train. Uncheck 3 seasons",
            options=season_options,
            )

        loaded_data = load_data(league=championship)
        train_data = deepcopy(loaded_data[loaded_data['season'].isin(training_seasons)]).reset_index(drop=True)
        validation_data = deepcopy(loaded_data[~loaded_data['season'].isin(training_seasons)]).reset_index(drop=True)
        # function below MUST BE COMPLETED
        model = train_model(championship=championship,
                            model_type=model_type_option,
                            train_data=preprocess(data_df=train_data,
                                                  model_type=model_type_option,
                                                  breaking_leg=break_leg),
                            validation_data=preprocess(data_df=validation_data,
                                                       model_type=model_type_option,
                                                       breaking_leg=break_leg)
                            )

    # provide the input
    # -- show the head of expected input dataframe
    sample_example_df = load_data(league='premier-league')
    st.dataframe(data=sample_example_df.head(), height=500, width=800)
    input_data = st.file_uploader(
        label="""\n\nProvide your input : a csv file with the above format collecting all games played 
        in one season until a specified leg.""")
    # input_path = st.text_input(label="Enter the path to you csv file.",
    # value="")
    start_prediction = st.button("Go for the prediction")
    if input_data is not None:
        if start_prediction:
            # Naive model
            st.markdown("#### Naive Prediction")
            # show prediction

            st.markdown(f"#### \n{model_type_option} model Prediction")
            # model.predict(data=input_data)
