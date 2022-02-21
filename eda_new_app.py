import os
import pandas as pd
import streamlit as st
from copy import deepcopy
# Custom modules

from eda.rank_based_eda import plot_team_pts_evol_vs_history
from preprocess.soccer_data import prepare_data, get_final_rank_performance_evolution

season_options = ['{start_year}-{end_year}'.format(start_year=year, end_year=year+1) for year in range(2004, 2019)]

championship_csv = {'ligue-1': 'ligue-1_data_2002_2019',
                    'ligue-2': 'ligue-2_data_2002_2019',
                    'serie-A': 'serie-a_data_2004_2019',
                    'bundesliga': 'bundesliga_data_2004_2019',
                    'premier-league': 'premier-league_data_2004_2019',
                    'liga': 'liga_data_2004_2019'}

SEASONS = [f"{year}-{year+1}" for year in range(2004, 2019)]

OPTIONS = {'0. Not interested :(': 0,
           """ 1.  Based on the final ranking, are you interested to see the evolution 
               from one the following kpis : cum_pts, cum_goal_diff, cum_goals_scored, goals_conceded, goals_scored, 
               rank ?""": 1,
           """ 2.  Do you to want to see how your team performs wrt to the average 
               evolution from another one (which must have played at least 5 seasons) ?""": 2,
           """ 3.  Do you want to see how a team perform in one season compared to its own 
           history ?""": 3,
           """ 4.  Do you want to see how a team is performing compared to the final 
           rank's requirements ?""": 4,
           """ 5.  Do you want to see an EDA on goal scoring performance ?""": 5
           }

KPI_TRANSLATOR = {'points': 'cum_pts', 'goal differences': 'cum_goal_diff', 'goals scored': 'cum_goals_scored'}


@st.cache(allow_output_mutation=True)
def load_data(league: str, raw: bool = True):
    """extract data"""
    csv_file = championship_csv.get(league, 'ligue-1_data_2002_2019')
    csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    return prepare_data(csv_path=csv_path, raw=raw)


@st.cache(allow_output_mutation=True)
def load_input_data(input_csv: str):
    """extract data"""
    # csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    df = prepare_data(csv_path=input_csv, raw=False)
    df = deepcopy(df.dropna(subset=['goals_scored'])).reset_index(drop=True)
    return df


@st.cache
def get_final_ranking_performance(data_df: pd.DataFrame):
    return get_final_rank_performance_evolution(data=data_df)


def app():
    # st.title('Soccer : what is the final ranking  ss?')
    # st.markdown(" Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")
    placeholder = st.empty()
    placeholder.markdown("### Exploratory Data Analysis with your new input")
    championship_choice = st.sidebar.selectbox(label="Select the championship you want to see",
                                               options=championship_csv.keys())
    # get data
    # championship_data = {champ: load_data(league=champ) for champ in championship_choices_list}
    sample_example_df = load_data(league='premier-league', raw=True)
    st.dataframe(data=sample_example_df.head(), height=500, width=800)

    championship_data = load_data(league=championship_choice, raw=False)

    st.markdown("##### Example of expected input")
    input_data = st.file_uploader(
        label="""\n\nProvide your input : a csv file with the above format collecting all games played 
                        in one season until a specified leg.""")
    # input_path = st.text_input(label="Enter the path to you csv file.",
    # value="")
    input_df = None
    team = ''
    show_std = False
    if input_data is not None:
        input_df = load_input_data(input_csv=input_data)
        team = st.selectbox(label="Please choose your team",
                            options=input_df.sort_values('team').team.unique())

        show_std = st.selectbox(label="Do you want the standard deviation area to be shown ?",
                                options=['yes', 'no'], index=1)
        show_std = show_std == 'yes'

    # show_trends = st.button("Show for the trends")
    if input_data is not None:
        # --> plot_compare_team_pts_evolution_vs_final_rank
        kpi = st.selectbox(label="What kpi to you to see the evolution ?",
                           options=['points', 'goal differences', 'goals scored'], index=0)
        st.plotly_chart(figure_or_data=plot_team_pts_evol_vs_history(history_df=championship_data,
                                                                     df=input_df,
                                                                     team=team,
                                                                     cum_points_col=KPI_TRANSLATOR[kpi],
                                                                     show_standard_deviation=show_std)
                        )
