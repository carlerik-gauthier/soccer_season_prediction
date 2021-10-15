import numpy as np
import pandas as pd

from copy import deepcopy

from rank_predictor.utils import get_lr_parameters


def get_season_team_data(break_leg: int, data: pd.DataFrame):
    """
    break_leg : split the season in two parts at this leg
    data: contains the season pts evolution wrt to legs for 1 team and 1 season
    
    returns: the Linear Regression parameters for both parts
    """
    data = data.reset_index(drop=True)
    
    breaking_cum_goal_diff = data.loc[break_leg-1, 'cum_goal_diff']
    breaking_cum_goal_scored = data.loc[break_leg-1, 'cum_goals_scored']
    roll_trend = data.loc[break_leg-1, 'rolling_5_games_avg_nb_points']
    trend = data.loc[break_leg-1,'avg_cum_pts_since_season_start']
    
    train_data = deepcopy(data[data.leg <= break_leg])
    eval_data = deepcopy(data[data.leg > break_leg])
    
    pts_at_break = data.loc[break_leg-1, 'cum_pts']
    eval_data.cum_pts -= pts_at_break
    eval_data.leg -= break_leg
    
    coef_feat, r_score_feat = get_lr_parameters(data=train_data)
    coef_predict, r_score_predict = get_lr_parameters(data=eval_data) if len(eval_data) > 0 else (-10, -10)
    
    final_nb_pts = data.loc[len(data)-1, 'cum_pts']
    
    nb_games_at_home = len(eval_data[eval_data.play=='Home'])
    
    return [nb_games_at_home, coef_feat, coef_predict, pts_at_break, final_nb_pts, 
            r_score_feat, r_score_predict, breaking_cum_goal_diff, breaking_cum_goal_scored,
            roll_trend, trend]


def build_data(historical_data: pd.DataFrame, break_leg: int):
    season_team_all = historical_data[['season', 'team']].values
    data_for_model = []
    ids_data = []
    # season_value = {s : i for i, s in enumerate(np.sort(df.season.unique()), start=1)}
    for season_team in np.unique(['###'.join(ll) for ll in season_team_all]):
        season, team = season_team.split("###")
        
        evol_feat = get_season_team_data(
            break_leg=break_leg, 
            data=deepcopy(historical_data[(historical_data.season==season) & (historical_data.team==team)]
                         )
        )
        data_for_model.append(evol_feat) # [season_value[season]]+evol_feat)
        ids_data.append([season, team])
    
    data_df = pd.DataFrame(columns=['nb_games_to_play_at_home',
                                    'lr_feat_coeff', 'lr_predict_coeff',
                                    'nb_pts_at_break', 'final_nb_pts', 
                                    'r_score_feat', 'r_score_predict', 
                                    'cumulative_goal_diff_at_break', 
                                    'cumulative_goal_scored_at_break', 
                                    'rolling_5_avg_pts_at_break', 
                                    'season_trend_at_break'
                                    ], 
                             data=np.array(data_for_model)
                            )
    
    ids_df = pd.DataFrame(columns=['season', 'team'], data=np.array(ids_data))

    return pd.concat([ids_df, data_df], axis=1)
    
    
    
def get_pivoted(data: pd.DataFrame, break_leg: int, value_col: str = 'cum_pts'):
    
    df = deepcopy(data[data.leg <= break_leg])
    df.rolling_5_games_avg_nb_points = [y if x!=x else x for x, y in 
           zip(df.rolling_5_games_avg_nb_points, df.avg_cum_pts_since_season_start)]
    
    df_pivot = df.pivot_table(index=['season', 'team'], 
                              columns='leg', 
                              values=[value_col]).reset_index()
    
    df_pivot.columns = [f'leg_{l}' if l!='' else n for n, l in df_pivot.columns]
    
    final = df[['season', 'team', 'final_rank', 'final_cum_pts']].drop_duplicates()
    df_last_leg = df[df.leg==break_leg][['season', 
                                         'team', 
                                         'rank', 
                                         'rolling_5_games_avg_nb_points', 
                                         'avg_cum_pts_since_season_start', 
                                         'cum_pts', 
                                         'cum_goal_diff', 
                                         'cum_goals_scored']].reset_index(drop=True)
    
    df_pivot = df_pivot.merge(df_last_leg, on=['season', 'team'])
    
    return df_pivot.merge(final, on=['season', 'team']).rename(columns={'final_cum_pts': 'final_nb_pts'})