import os
import pandas as pd
import streamlit as st

# Custom modules
import eda.basics as edab

from eda.goals_related_eda import hist_aggregator
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

SEASONS = [f"{year}-{year+1}" for year in range(2004, 2019)]


@st.cache
def load_data(league: str, raw: bool = True):
    """extract data"""
    csv_file = championship_csv.get(league, 'ligue-1_data_2002_2019')
    csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    return prepare_data(csv_path=csv_path, raw=raw)


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


st.markdown("#### Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")
placeholder = st.empty()


def app():
    placeholder.markdown("# Exploratory Data Analysis")
    championship_choice = st.sidebar.selectbox(label="Select the championship you want to see",
                                               options=championship_csv.keys())
    # get data
    # championship_data = {champ: load_data(league=champ) for champ in championship_choices_list}
    championship_data = load_data(league=championship_choice, raw=False)
    # show basic eda
    st.markdown("## Basic EDA")
    col1, space1, col2 = st.columns((10, 3, 10))
    participation_df = edab.get_team_participation(df=championship_data)
    st.write(f"""{championship_data.team.nunique()} teams have played in {championship_choice} 
        from season {championship_data.season.min()} to season {championship_data.season.max()}, 
        i.e over {championship_data.season.nunique()} seasons. \n By descending order, one has : """)
    _, col3, _ = st.columns((4, 20, 4))
    # team participation : get_team_participation
    # Home-Away effect : nb points and goal scored -- hist_aggregator
    # Leg effect : nb points and goals scored -- hist_aggregator
    with col1:
        # Home-Away
        st.markdown("#### Home-Away Benefit on team performance")
        home_pts = hist_aggregator(df=championship_data[championship_data.play == 'Home'],
                                   column_to_describe='nb_points',
                                   aggreg_column='play')
        st.dataframe(data=home_pts)
        st.markdown("#### Home-Away Benefit on the number of goals the team scores")
        home_away_goals_scored = hist_aggregator(df=championship_data,
                                                 column_to_describe='goals_scored',
                                                 aggreg_column='play')
        st.dataframe(data=home_away_goals_scored)
    with col2:
        # Leg
        st.markdown("#### Leg Benefit on team performance")
        leg_on_perf_at_home = hist_aggregator(df=championship_data[championship_data.play == 'Home'],
                                              column_to_describe='nb_points',
                                              aggreg_column='leg')
        st.dataframe(data=leg_on_perf_at_home)

        st.markdown("#### Leg Benefit on the number of goals the team scores")

        leg_goals = hist_aggregator(df=championship_data, column_to_describe='goals_scored', aggreg_column='leg')
        st.dataframe(data=leg_goals)

    with col3:

        st.dataframe(data=participation_df)

        st.write("\n {nb_all_seasons} teams played all {nb_seasons} seasons".format(
            nb_all_seasons=len(participation_df[participation_df.nb_participation == championship_data.season.nunique()]),
            nb_seasons=championship_data.season.nunique())
        )
    # show eda plots
    # How does a team perform in one season compared to its own history ?
    # compare_pts_evol_time
    #
    # How is a team performing compared to the final rank's requirements ?
    # plot_compare_team_pts_evolution_vs_final_rank
    placeholder_1b = st.markdown("# EDA questions according to general ranking performances")
    placeholder_1b2 = st.write(
        "Please answer 'yes' to one of the EDA question in the sidebar to go further with the EDA")
    option_1 = st.sidebar.selectbox(label="""   Based on the final ranking, are you interested to see the evolution 
    from one the following kpis : cum_pts, cum_goal_diff, cum_goals_scored, goals_conceded, goals_scored,  rank ?""",
                                    options=['yes', 'no'], index=1)
    if option_1 == 'yes':
        placeholder_1b2.empty()
        kpi_choice = st.selectbox(label="Please choose the kpi :",
                                  options=['cum_pts',
                                           'cum_goal_diff',
                                           'cum_goals_scored',
                                           'goals_conceded',
                                           'goals_scored',
                                           'rank']
                                  )
        # --> plot_kpi_evolution
        ...
    option_2 = st.sidebar.selectbox(label="""   Do you to want to see how your team performs wrt to the average 
    evolution from another one (which must have played at least 5 seasons) ?""", options=['yes', 'no'], index=1)
    if option_2 == 'yes':
        placeholder_1b2.empty()
        team = st.selectbox(label="Please choose your team",
                            options=...)
        season_selected = st.selectbox(label="Choose a specific season if wanted",
                                       options=['']+SEASONS)
        season_selected = None if season_selected == '' else season_selected
        team_to_compare = st.selectbox(label="Pick the team you want to compare with",
                                       options=...)
        # --> compare_pts_evol_with_avg_evolution
        ...
    option_3 = st.sidebar.selectbox(label="""   Do you want to see how a team perform in one season compared to its own 
    history ?""", options=['yes', 'no'], index=1)
    if option_3 == "yes":
        placeholder_1b2.empty()
        team = st.selectbox(label="Please choose your team",
                            options=...)
        # --> compare_pts_evol_time
        ...
    option_4 = st.sidebar.selectbox(label="""   Do you want to see how a team is performing compared to the final 
    rank's requirements ? ?""", options=['yes', 'no'], index=1)
    if option_4 == 'yes':
        placeholder_1b2.empty()
        team = st.selectbox(label="Please choose your team",
                            options=...)
        season_selected = st.selectbox(label="Choose a specific season if wanted",
                                       options=[''] + SEASONS)
        season_selected = None if season_selected == '' else season_selected
        show_std = st.selectbox(label="Do you want the standard deviation area to be shown ?",
                                options=['yes', 'no'], index=1)
        show_std = show_std == 'yes'
        # --> plot_compare_team_pts_evolution_vs_final_rank
        ...
    option_5 = st.sidebar.selectbox(label="""  Do you want to see an EDA on goal scoring performance ?""",
                                    options=['yes', 'no'], index=1)
    if option_5 == 'yes':
        placeholder_1b2.empty()
        st.write("EDA on goal scoring performance")
        # # Goals
        col3, space2, col4 = st.columns((10, 1, 10))
        with col3:
            # team
            ...
        with col4:
            # opponent
            ...
        #  Team  vs Opponent:
        #  Season average : Scored / opponent conceded
        #
        #  Average 5 last game : goals scored / opponent avg goals conceded on number goals scored
        #
        #  Last game performance
        #  Scored vs opponent conceded
        #
        #  Outcome : 5 rolling games
        #  Outcome : Last game
