import os
import pandas as pd
import streamlit as st
from copy import deepcopy

# Custom modules

from utils import is_available, train_model, retrieve_model, get_model_performance
from preprocess.soccer_data import prepare_data
from preprocess.predictor_preprocess import build_data, get_pivoted


# TODO : 1. connection to model

season_options = ['{start_year}-{end_year}'.format(start_year=year, end_year=year+1) for year in range(2004, 2019)]

championship_csv = {'ligue-1': 'ligue-1_data_2002_2019',
                    'ligue-2': 'ligue-2_data_2002_2019',
                    'serie-A': 'serie-a_data_2004_2019',
                    'bundesliga': 'bundesliga_data_2004_2019',
                    'premier-league': 'premier-league_data_2004_2019',
                    'liga': 'liga_data_2004_2019'}

SEASONS = [f"{year}-{year+1}" for year in range(2004, 2019)]


@st.cache
def load_data(league: str, raw: bool = True):
    """extract data"""
    csv_file = championship_csv.get(league, 'ligue-1_data_2002_2019')
    csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    return prepare_data(csv_path=csv_path, raw=raw)


@st.cache
def extract_final_perf(data_df: pd.DataFrame,
                       final_rank_col: str = 'final_rank',
                       final_nb_points: str = 'final_cum_pts'):
    dg = data_df[['season', 'team', final_rank_col, final_nb_points]].drop_duplicates().reset_index(drop=True)
    return dg


@st.cache
def preprocess(data_df: pd.DataFrame, model_type: str = 'naive', breaking_leg: int = 27):
    """Preprocess data according to the choice of the model"""
    if model_type == 'classification':
        df = get_pivoted(data=data_df, break_leg=breaking_leg)
    elif model_type == 'regression':
        df = build_data(historical_data=data_df, break_leg=breaking_leg)
    elif model_type == 'ranking':
        df = get_pivoted(data=data_df, break_leg=breaking_leg)
    else:
        # will process Naive model
        df = build_data(historical_data=data_df, break_leg=breaking_leg)

    return df.sort_values(by='season').reset_index(drop=True)


def app():
    # st.title('Soccer : what is the final ranking ?')

    # st.markdown(" Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")
    placeholder = st.empty()

    placeholder.markdown("##### Predict the ranking")

    st.sidebar.markdown("#### Naive model is included")
    # choose championship
    championship = st.sidebar.selectbox(label="Choose your championship you want to predict on",
                                        options=championship_csv.keys())
    # choose the model type : regression, classification or ranking algorithm
    model_type_option = st.sidebar.selectbox(
        label="Please choose the type of algorithm used by the ranker.",
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
    has_model = False
    if model_available:
        choice = st.sidebar.selectbox(label="Do you want to use an already trained model ?", options=['yes', 'no'])
        use_pretrained = choice == 'yes'
    # -- retrieve a pretrained model if requested and available else train the model
    if use_pretrained:
        model = retrieve_model(module_path="saved_models", file_name=model_name)
        has_model = True
    else:
        # train model
        form = st.sidebar.form(key="training seasons")
        key = 0
        training_seasons = form.multiselect(
            label="Select the seasons on which the model should train. Uncheck 3 seasons",
            options=season_options,
            default=season_options,
            key=key
        )
        submit = form.form_submit_button(label="Go for training !!!")
        if submit and len(season_options)-len(training_seasons) != 3:
            st.write("""{val_length} seasons have been unchecked instead of the required 3.
                 Select the seasons on which the model should train. Uncheck 3 seasons.
                 """.format(val_length=len(season_options)-len(training_seasons)))
        elif submit:
            loaded_data = load_data(league=championship, raw=False)
            train_data = deepcopy(loaded_data[loaded_data['season'].isin(training_seasons)]).reset_index(drop=True)
            validation_data = deepcopy(loaded_data[~loaded_data['season'].isin(training_seasons)]).reset_index(
                drop=True)

            preprocessed_train_data = preprocess(data_df=train_data,
                                                 model_type=model_type_option,
                                                 breaking_leg=break_leg)
            st.dataframe(preprocessed_train_data)
            model = train_model(championship=championship,
                                model_type=model_type_option,
                                nb_opponent=18 if championship == 'bundesliga' else 20,
                                train_data=preprocessed_train_data,
                                model_name=model_name
                                )
            has_model = True
            preprocessed_validation_data = preprocess(data_df=validation_data,
                                                      model_type=model_type_option,
                                                      breaking_leg=break_leg)
            if model_type_option == 'regression':
                final_perf_validation = extract_final_perf(data_df=validation_data)
                preprocessed_validation_data = pd.merge(final_perf_validation, preprocessed_validation_data, how='left',
                                                        on=['season', 'team'])
            placeholder2 = st.empty()
            placeholder2.write("getting model performance ...")
            perf = get_model_performance(test_data=preprocessed_validation_data,
                                         model=model
                                         )
            placeholder2.write(f"Model Training Performance is {perf}%")
    if has_model:
        predict_button = st.button("Let's predict !!!")
        # provide the input
        # -- show the head of expected input dataframe
        if predict_button:
            st.markdown("##### Example of expected input")
            sample_example_df = load_data(league='premier-league', raw=True)
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
