import os
import pandas as pd
import streamlit as st
from copy import deepcopy
# Custom modules
import eda.basics as edab

from eda.goals_related_eda import hist_aggregator
from eda.rank_based_eda import plot_kpi_evolution, plot_team_pts_evol_with_competitor_avg_evolution, \
    plot_team_pts_evol_to_average_performance, plot_team_pts_evol_vs_final_rank
from eda.utils import draw_line, draw_pie_chart, draw_sunburst
from eda.goals_related_eda import mean_aggregator
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


@st.cache(allow_output_mutation=True)
def load_data(league: str, raw: bool = True):
    """extract data"""
    csv_file = championship_csv.get(league, 'ligue-1_data_2002_2019')
    csv_path = os.path.join(os.path.dirname(__file__), os.path.join('inputs', csv_file))
    return prepare_data(csv_path=csv_path, raw=raw)


@st.cache
def get_final_ranking_performance(data_df: pd.DataFrame):
    return get_final_rank_performance_evolution(data=data_df)


def app():
    # st.title('Soccer : what is the final ranking  ss?')
    # st.markdown(" Data comes from l'Équipe website and runs from season 2004-2005 to 2018-2019")
    placeholder = st.empty()
    placeholder.markdown("### Exploratory Data Analysis")
    championship_choice = st.sidebar.selectbox(label="Select the championship you want to see",
                                               options=championship_csv.keys())
    # get data
    # championship_data = {champ: load_data(league=champ) for champ in championship_choices_list}
    championship_data = load_data(league=championship_choice, raw=False)
    championship_data_final_rank_df = get_final_ranking_performance(data_df=championship_data)
    # show basic eda
    st.markdown("## Basic EDA")
    col1, space1, col2 = st.columns((20, 10, 20))
    participation_df = edab.get_team_participation(df=championship_data)
    st.write(f"""{championship_data.team.nunique()} teams have played in {championship_choice} 
        from season {championship_data.season.min()} to season {championship_data.season.max()}, 
        i.e over {championship_data.season.nunique()} seasons. \n By descending order, one has : """)
    _, col3, _ = st.columns((4, 20, 4))
    # team participation : get_team_participation
    # Home-Away effect : nb points and goal scored -- hist_aggregator
    # Leg effect : nb points and goals scored -- hist_aggregator
    st.markdown("#### Home-Away Benefit on team performance")
    st.write("Please expand the figures for the details 😉")
    home_pts = hist_aggregator(df=championship_data[championship_data.play == 'Home'],
                               column_to_describe='nb_points',
                               aggreg_column='play')
    st.dataframe(data=home_pts)
    st.plotly_chart(figure_or_data=draw_pie_chart(
        df=home_pts,
        values='cnt',
        names='nb_points',
        title='Home performance'),
        use_container_width=True
    )
    # with col1:
    #     # Home-Away
    #     st.markdown("#### Home-Away Benefit on team performance")
    #     st.write("Please expand the figures for the details 😉")
    #     home_pts = hist_aggregator(df=championship_data[championship_data.play == 'Home'],
    #                                column_to_describe='nb_points',
    #                                aggreg_column='play')
    #     st.dataframe(data=home_pts)
    #     st.plotly_chart(figure_or_data=draw_pie_chart(
    #         df=home_pts,
    #         values='cnt',
    #         names='nb_points',
    #         title='Home performance'),
    #         use_container_width=True
    #     )
    #     st.markdown("#### Home-Away Benefit on the number of goals the team scores")
    #     home_away_goals_scored = hist_aggregator(df=championship_data,
    #                                              column_to_describe='goals_scored',
    #                                              aggreg_column='play')
    #     st.dataframe(data=home_away_goals_scored)
    #     st.plotly_chart(figure_or_data=draw_sunburst(
    #         df=home_away_goals_scored,
    #         values='cnt',
    #         path=['play', 'goals_scored']),
    #         use_container_width=True
    #     )
    # with col2:
    #     # Leg
    #     st.markdown("#### Leg Benefit on team performance")
    #     leg_on_perf_at_home = hist_aggregator(df=championship_data[championship_data.play == 'Home'],
    #                                           column_to_describe='nb_points',
    #                                           aggreg_column='leg')
    #     st.dataframe(data=leg_on_perf_at_home)
    #     st.plotly_chart(figure_or_data=draw_sunburst(df=leg_on_perf_at_home,
    #                                                  path=['leg', 'nb_points'],
    #                                                  values='cnt'),
    #                     use_container_width=True
    #                     )
    #     st.markdown("#### Leg Benefit on the number of goals the team scores")
    #
    #     leg_goals = hist_aggregator(df=championship_data, column_to_describe='goals_scored', aggreg_column='leg')
    #     st.dataframe(data=leg_goals)
    #     st.plotly_chart(figure_or_data=draw_sunburst(df=leg_goals,
    #                     path=['leg', 'goals_scored'],
    #                     values='cnt'),
    #                     use_container_width=True
    #                     )

    with col3:

        st.dataframe(data=participation_df)

        st.write("""\n {nb_all_seasons} teams played all {nb_seasons} seasons. There have been {nb_champion} different
        champions""".format(
            nb_all_seasons=len(participation_df[participation_df.nb_participation == championship_data.season.nunique()]
                               ),
            nb_seasons=championship_data.season.nunique(),
            nb_champion=len(participation_df[participation_df.nb_titles > 0]))
        )
    # show eda plots
    # How does a team perform in one season compared to its own history ?
    # compare_pts_evol_time
    #
    # How is a team performing compared to the final rank's requirements ?
    # plot_compare_team_pts_evolution_vs_final_rank
    st.markdown("# EDA questions according to general ranking performances")
    st.write("Please answer 'yes' to one of the EDA question in the sidebar to go further with the EDA")
    eda_option = st.sidebar.radio(label="Please select of the EDA question in the sidebar to go further with the EDA",
                                  options=OPTIONS.keys())
    if OPTIONS[eda_option] == 1:
        # placeholder_1b2.empty()
        kpi_choice = st.selectbox(label="Please choose the kpi :",
                                  options=['cum_pts',
                                           'cum_goal_diff',
                                           'cum_goals_scored',
                                           'goals_conceded',
                                           'goals_scored',
                                           'rank']
                                  )
        std_choice = st.selectbox(label="Do you want to see the standard deviation shadow ?",
                                  options=['yes', 'no']
                                  )
        # --> plot_kpi_evolution
        st.plotly_chart(figure_or_data=plot_kpi_evolution(df=championship_data_final_rank_df,
                                                          kpi=kpi_choice,
                                                          show_standard_deviation=std_choice == 'yes')
                        )

    if OPTIONS[eda_option] == 2:
        # placeholder_1b2.empty()
        season_selected = st.selectbox(label="Choose a specific season if wanted",
                                       options=SEASONS)
        # season_selected = None if season_selected == '' else season_selected

        team_df = deepcopy(championship_data[championship_data.season == season_selected]) if season_selected else \
            deepcopy(championship_data)
        team = st.selectbox(label="Please choose your team",
                            options=team_df.sort_values('team').team.unique())

        team_to_compare = st.selectbox(label="Pick the team you want to compare with",
                                       options=participation_df[participation_df.nb_participation >= 5].index)
        # --> compare_pts_evol_with_avg_evolution
        st.plotly_chart(figure_or_data=plot_team_pts_evol_with_competitor_avg_evolution(
            data=championship_data,
            team=team,
            season=season_selected,
            compare_with=team_to_compare)
        )
    if OPTIONS[eda_option] == 3:
        # placeholder_1b2.empty()
        team = st.selectbox(label="Please choose your team",
                            options=participation_df[participation_df.nb_participation >= 3].index)
        # --> compare_pts_evol_time
        st.plotly_chart(figure_or_data=plot_team_pts_evol_to_average_performance(
            data=championship_data,
            team=team)
        )
    if OPTIONS[eda_option] == 4:
        # placeholder_1b2.empty()
        season_selected = st.selectbox(label="Choose a specific season if wanted",
                                       options=[''] + SEASONS)
        season_selected = None if season_selected == '' else season_selected

        team_df = deepcopy(championship_data[championship_data.season == season_selected]) if season_selected else \
            deepcopy(championship_data)

        team = st.selectbox(label="Please choose your team",
                            options=team_df.sort_values('team').team.unique())
        show_std = st.selectbox(label="Do you want the standard deviation area to be shown ?",
                                options=['yes', 'no'], index=1)
        show_std = show_std == 'yes'
        if season_selected:
            tmp_df = deepcopy(team_df[team_df.team == team]).reset_index(drop=True)
            final_rank = tmp_df.loc[0, 'final_rank']
            st.text(f"{team}'s final rank for season {season_selected}: {final_rank}")
        # --> plot_compare_team_pts_evolution_vs_final_rank
        st.plotly_chart(figure_or_data=plot_team_pts_evol_vs_final_rank(df=championship_data,
                                                                        team=team,
                                                                        season=season_selected,
                                                                        show_standard_deviation=show_std)
                        )
    if OPTIONS[eda_option] == 5:
        # placeholder_1b2.empty()
        st.markdown("#### EDA on goal scoring performance")
        st.write("Do not hesitate to expand the figures 🔍")
        # # Goals
        col3, space2, col4 = st.columns((20, 5, 20))
        with col3:
            st.markdown("## Team")
            st.markdown("#### Average number of goals scored by the team in the season so far vs goals to be scored")
            team_season_perf_on_goals_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_team_avg_goals_scored_since_season_start',
                bin_step=.1)
            st.plotly_chart(figure_or_data=draw_line(
                df=team_season_perf_on_goals_mean,
                x='previous_team_avg_goals_scored_since_season_start_binned',
                y='avg_goals_scored',
                title='Team avg goals scored since season start vs avg goals to be scored'),
                use_container_width=True
            )

            st.markdown("#### Average number of goals scored by the team in the last 5 games vs goals to be scored")
            team_last5_perf_on_goals_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_team_rolling_5_games_avg_goals_scored',
                bin_step=.1)

            st.plotly_chart(figure_or_data=draw_line(
                df=team_last5_perf_on_goals_mean,
                x='previous_team_rolling_5_games_avg_goals_scored_binned',
                y='avg_goals_scored',
                title='5 leg Avg on Team goals scored vs avg goals to be scored'),
                use_container_width=True
            )
            st.markdown("#### Last game number of goals scored by the team vs goals to be scored")
            last_game_team_goals_scored_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_team_goals_scored',
                bin_step=None)

            st.plotly_chart(figure_or_data=draw_line(
                df=last_game_team_goals_scored_mean,
                x='previous_team_goals_scored',
                y='avg_goals_scored',
                title='Team previous game goals scored vs avg goals to be scored'),
                use_container_width=True
            )

            st.markdown("#### Average number of points won by the team in the last 5 games vs goals to be scored")
            last_5games_team_outcome_mean = mean_aggregator(df=championship_data,
                                                            column_to_describe='goals_scored',
                                                            aggreg_column='previous_team_rolling_5_games_avg_nb_points',
                                                            bin_step=None)
            st.plotly_chart(figure_or_data=draw_line(
                df=last_5games_team_outcome_mean,
                x='previous_team_rolling_5_games_avg_nb_points',
                y='avg_goals_scored',
                title='Team last 5 games outcome conceded vs avg goals to be scored'),
                use_container_width=True
            )
            st.markdown("#### Last game number of points won by the team vs goals to be scored")
            last_game_team_outcome_mean = mean_aggregator(df=championship_data,
                                                          column_to_describe='goals_scored',
                                                          aggreg_column='previous_team_nb_points',
                                                          bin_step=None)
            st.plotly_chart(figure_or_data=draw_line(
                df=last_game_team_outcome_mean,
                x='previous_team_nb_points',
                y='avg_goals_scored',
                title='Team game outcome conceded vs avg goals to be scored'),
                use_container_width=True
            )
        with col4:
            # opponent
            st.markdown("## Opponent")
            st.markdown(
                """#### Average number of goals conceded by the opponent in the season so far vs goals to be scored""")
            opponent_season_perf_on_goals_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_opponent_avg_goals_conceded_since_season_start',
                bin_step=.1)

            st.plotly_chart(figure_or_data=draw_line(
                df=opponent_season_perf_on_goals_mean,
                x='previous_opponent_avg_goals_conceded_since_season_start_binned',
                y='avg_goals_scored',
                title='Opponent avg goals conceded since season start vs avg goals to be scored'),
                use_container_width=True
            )
            st.markdown(
                """#### Average number of goals conceded by the opponent in the last 5 games vs goals to be scored""")
            opponent_last5_perf_on_goals_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_opponent_rolling_5_games_avg_goals_conceded',
                bin_step=.1)
            st.plotly_chart(figure_or_data=draw_line(
                df=opponent_last5_perf_on_goals_mean,
                x='previous_opponent_rolling_5_games_avg_goals_conceded_binned',
                y='avg_goals_scored',
                title='5 leg Avg on opponents goals conceded vs avg goals to be scored'),
                use_container_width=True
            )
            st.markdown("#### Last game number of goals conceded by the opponent vs goals to be scored")
            last_game_opponent_goals_conceded_mean = mean_aggregator(df=championship_data,
                                                                     column_to_describe='goals_scored',
                                                                     aggreg_column='previous_opponent_goals_conceded',
                                                                     bin_step=None)

            st.plotly_chart(figure_or_data=draw_line(
                df=last_game_opponent_goals_conceded_mean,
                x='previous_opponent_goals_conceded',
                y='avg_goals_scored',
                title='Opponent previous game goals conceded vs avg goals to be scored'),
                use_container_width=True
            )

            st.markdown("""#### Average number of of points won by the opponent in the last 5 games vs goals to be 
            scored""")
            last_5games_opponent_outcome_mean = mean_aggregator(
                df=championship_data,
                column_to_describe='goals_scored',
                aggreg_column='previous_opponent_rolling_5_games_avg_nb_points',
                bin_step=None)
            st.plotly_chart(figure_or_data=draw_line(
                df=last_5games_opponent_outcome_mean,
                x='previous_opponent_rolling_5_games_avg_nb_points',
                y='avg_goals_scored',
                title='Opponent last 5 games outcome conceded vs avg goals to be scored'),
                use_container_width=True
            )
            st.markdown("#### Last game number of points won by the opponent vs goals to be scored")
            last_game_opponent_outcome_mean = mean_aggregator(df=championship_data,
                                                              column_to_describe='goals_scored',
                                                              aggreg_column='previous_opponent_nb_points',
                                                              bin_step=None)
            st.plotly_chart(figure_or_data=draw_line(
                df=last_game_opponent_outcome_mean,
                x='previous_opponent_nb_points',
                y='avg_goals_scored',
                title='Opponent game outcome conceded vs avg goals to be scored'),
                use_container_width=True
            )
