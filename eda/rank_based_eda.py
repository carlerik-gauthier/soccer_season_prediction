import plotly.graph_objects as go
import plotly.express as px

from copy import deepcopy

# plot_plotly_kpi
def plot_kpi_evolution(df, kpi='cum_pts', show_standard_deviation=False):
    admissible_kpis = {'rank', 
                       'cum_pts', 
                       'cum_goal_diff', 
                       'cum_goals_scored',
                       'goals_conceded', 
                       'goals_scored'
                      }
    yaxis_title_dict = {'cum_pts': 'Number of points',
                        'rank': 'Rank',
                        'cum_goal_diff': 'Goal difference',
                        'cum_goals_scored': 'Goal scored',
                        'goals_conceded': 'Goal conceded',
                        'goals_scored': 'Goal scored'
                       }
    
    avg_col = f'avg_{kpi}'
    std_col = f'std_{kpi}'
    
    if kpi not in admissible_kpis:
        raise Exception(f"""
        kpi {kpi} is not admissible. It must be part of the following set : {admissible_kpis}
        """)
    
    if not show_standard_deviation:
        fig = px.line(data_frame=df, x="leg", y=avg_col, color="final_rank",
              title=f"Average {kpi} Evolution based on final ranking",
             )

        fig.update_layout(
            autosize=False,
            width=800,
            height=800)

        fig.show()
    else:
        go_layers = []
        for ranking in df.final_rank.unique()[::-1]:
            dg = df[df.final_rank == ranking]
            sublayer = [
            go.Scatter(
                name=str(ranking),
                x=dg['leg'],
                y=dg[avg_col],
                mode='lines',
                line=dict(color=color_2_position[ranking],
                         width=2 if ranking not in [3, 18] else 5)
                    ),

            go.Scatter(
                name=f'Upper Bound {ranking}',
                x=dg['leg'],
                y=dg[avg_col]+dg[std_col],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
                    ),

            go.Scatter(
                name=f'Lower Bound {ranking}',
                x=dg['leg'],
                y=dg[avg_col]-dg[std_col],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor=color_name_to_rgba(name=color_2_position[ranking], fill=0.1),
                fill='tonexty',
                showlegend=False,
            )
            ]
            go_layers+=sublayer



        layout = go.Layout(
            autosize=True, #False,
            width=800,
            height=800,

            xaxis= go.layout.XAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            yaxis= go.layout.YAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad = 4
            )
        )

        fig = go.Figure(data=go_layers, layout=layout)

        fig.update_layout(
            yaxis_title=yaxis_title_dict[kpi],
            xaxis_title='leg',
            title=f"{kpi} evolution according to final ranking",
            hovermode="x"
        )

        #fig.update_layout(
        #    autosize=False,
        #    width=800,
        #    height=800)

        fig.show()


def compare_pts_evol_with_avg_evolution(data, team, season=None, until_leg=38, compare_with=None):
    """
    :param data: pd.DataFrame: data containing the league performance
    :param team: str: name of the team we want to analyze
    :param season: str: season we're interested in
    :param until_leg: int: plot team's pts evolution from legs 1 to until leg included
    :param compare_with: str: name of the team whose average pts evolution is computed and which is used for 
    comparison. That Team MUST have played at least 5 seasons
    """
    if season is None:
        team_data = deepcopy(data[(data.team==team) & (data.leg <= until_leg)])
    else:
        team_data = deepcopy(data[(data.team == team) & (data.season==season) & (data.leg <= until_leg)])
    comparator_data = deepcopy(data[data.team==compare_with])
    
    nb_season = comparator_data.season.nunique()
    if nb_season < 4 or len(team_data)==0:
        raise ValueError(f"""{team} has not played season {season} or {comparator_data} has played at most 
                         4 games. Please review your inputs""")
        
    avg_comparator_data = comparator_data[['leg', 'cum_pts']].groupby(
        by=['leg']).mean().reset_index().rename(columns={'cum_pts':'avg_cum_pts'})
    
    go_layers= [
    go.Scatter(name=f"{compare_with} averaged",
               x=avg_comparator_data['leg'],
               y=avg_comparator_data['avg_cum_pts'],
               mode='lines',
               line=dict(color="red",
                        width=5)
                   ),
     go.Scatter(name=team,
                x=team_data['leg'],
                y=team_data['cum_pts'],
                mode='lines',
                line=dict(color="royalblue",
                          width=2)
                 )
    ]
    
    layout = go.Layout(
            autosize=True, #False,
            width=800,
            height=800,

            xaxis= go.layout.XAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            yaxis= go.layout.YAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad = 4
            )
        )

    fig = go.Figure(data=go_layers, layout=layout)

    fig.update_layout(
        yaxis_title="number of points",
        title=f"{team}'s point evolution during season {season} wrt to {compare_with} average pts evolution",
        hovermode="x"
    )

    #fig.update_layout(
    #    autosize=False,
    #    width=800,
    #    height=800)

    fig.show() 
    
def compare_pts_evol_time(data, team, until_leg=38):
    """
    :param data: pd.DataFrame: data containing the league performance
    :param team: str: name of the team we want to analyze
    :param until_leg: int: plot team's pts evolution from legs 1 to until leg included
    comparison. That Team MUST have played at least 5 seasons
    """
    team_data = deepcopy(data[(data.team == team) & (data.leg <= until_leg)])
    comparator_data = deepcopy(data[data.team==team])
    
    nb_season = comparator_data.season.nunique()
    if nb_season < 4:
        raise ValueError(f"""{compare_with} has played at most 4 games in season {season}.
        Please pick a team having played at least 5 seasons. """)
    if len(team_data)==0:
        raise ValueError(f"""{team} has not played season {season}. 
        Please pick a team having played at least 5 seasons """)
        
    avg_comparator_data = comparator_data[['leg', 'cum_pts']].groupby(
        by=['leg']).mean().reset_index().rename(columns={'cum_pts':'avg_cum_pts'})
    
    go_layers= [
    go.Scatter(name="averaged point evolution",
               x=avg_comparator_data['leg'],
               y=avg_comparator_data['avg_cum_pts'],
               mode='lines',
               line=dict(color="red",
                        width=5)
                   )]
    i = 0
    for season_start in range(2004,2019):
        i+=1
        season = f'{season_start}-{season_start+1}'
        sublayer = [
         go.Scatter(name=season,
                    x=team_data[team_data.season==season]['leg'],
                    y=team_data[team_data.season==season]['cum_pts'],
                    mode='lines',
                    line=dict(color=color_2_position[i],
                              width=2)
                     )
        ]
        
        go_layers+=sublayer
    
    layout = go.Layout(
            autosize=True, #False,
            width=800,
            height=800,

            xaxis= go.layout.XAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            yaxis= go.layout.YAxis(linecolor = 'black',
                                  linewidth = 1,
                                  mirror = True),

            margin=go.layout.Margin(
                l=50,
                r=50,
                b=100,
                t=100,
                pad = 4
            )
        )

    fig = go.Figure(data=go_layers, layout=layout)

    fig.update_layout(
        yaxis_title="number of points",
        title=f"{team}'s point evolution over its {nb_season} seasons wrt to its average pts evolution",
        hovermode="x"
    )

    #fig.update_layout(
    #    autosize=False,
    #    width=800,
    #    height=800)

    fig.show() 
    
def plot_compare_team_pts_evolution_vs_final_rank(df, team, season=None, show_standard_deviation=True):
    
    kpi='cum_pts'
    avg_col = f'avg_{kpi}'
    std_col = f'std_{kpi}'
        
    go_layers = []
    
    if season is None:
        tmp_df = deepcopy(df[df.team==team])
        team_df = tmp_df[['leg', 'cum_pts']].groupby(
        by=['leg']).mean().reset_index().rename(columns={'cum_pts':'avg_cum_pts'})
        
        sub_layer = [
                    go.Scatter(name=f"{team} averaged point evolution",
                               x=team_df['leg'],
                               y=team_df['avg_cum_pts'],
                               mode='lines',
                               line=dict(color="silver",
                                        width=6)
                   )]
    else:
        team_df = deepcopy(df[(df.team == team) & (df.season==season)])
        sub_layer = [
                    go.Scatter(name=f"{team} point evolution",
                               x=team_df['leg'],
                               y=team_df['cum_pts'],
                               mode='lines',
                               line=dict(color="gold",
                                        width=6)
                   )]
    
    comparator = df.groupby(by=['final_rank', 'leg']).aggregate({'cum_pts': ['mean', 'std']})
    comparator.columns = [avg_col, std_col]
    comp_final = comparator.reset_index()
    
    go_layers += sub_layer
    for ranking in comp_final.final_rank.unique()[::-1]:
        dg = comp_final[comp_final.final_rank == ranking]
        sublayer = [
        go.Scatter(
            name=str(ranking),
            x=dg['leg'],
            y=dg[avg_col],
            mode='lines',
            line=dict(color=color_2_position[ranking],
                     width=2)
                ),

        go.Scatter(
            name=f'Upper Bound {ranking}',
            x=dg['leg'],
            y=dg[avg_col]+dg[std_col],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
                ),

        go.Scatter(
            name=f'Lower Bound {ranking}',
            x=dg['leg'],
            y=dg[avg_col]-dg[std_col],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor=color_name_to_rgba(name=color_2_position[ranking], fill=0.1*show_standard_deviation),
            fill='tonexty',
            showlegend=False,
        )
        ]
        go_layers+=sublayer



    layout = go.Layout(
        autosize=True, #False,
        width=800,
        height=800,

        xaxis= go.layout.XAxis(linecolor = 'black',
                              linewidth = 1,
                              mirror = True),

        yaxis= go.layout.YAxis(linecolor = 'black',
                              linewidth = 1,
                              mirror = True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad = 4
        )
    )

    fig = go.Figure(data=go_layers, layout=layout)

    fig.update_layout(
        yaxis_title='Number of points',
        title=f"{kpi} Evolution according to final ranking",
        hovermode="x"
    )

    #fig.update_layout(
    #    autosize=False,
    #    width=800,
    #    height=800)

    fig.show()

