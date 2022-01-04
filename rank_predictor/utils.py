import pandas as pd
import numpy as np

from copy import deepcopy
from sklearn.linear_model import LinearRegression


def get_lr_parameters(data: pd.DataFrame):
    tmp_x_data = data.leg.values
    tmp_y_data = data.cum_pts.values
    
    x_data = np.array([0] + list(tmp_x_data)).reshape(-1, 1)
    y_data = np.array([0] + list(tmp_y_data)).reshape(-1, 1)
    
    reg = LinearRegression(fit_intercept=False).fit(X=x_data, y=y_data)
    
    return reg.coef_[0][0], reg.score(X=x_data, y=y_data)


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
    trend = data.loc[break_leg-1, 'avg_cum_pts_since_season_start']
    
    train_data = deepcopy(data[data.leg <= break_leg])
    eval_data = deepcopy(data[data.leg > break_leg])
    
    pts_at_break = data.loc[break_leg-1, 'cum_pts']
    eval_data.cum_pts -= pts_at_break
    eval_data.leg -= break_leg
    
    coef_feat, r_score_feat = get_lr_parameters(data=train_data)
    coef_predict, r_score_predict = get_lr_parameters(data=eval_data) if len(eval_data) > 0 else (-10, -10)
    
    final_nb_pts = data.loc[len(data)-1, 'cum_pts']
    
    nb_games_at_home = len(eval_data[eval_data.play == 'Home'])
    
    return [nb_games_at_home, coef_feat, coef_predict, pts_at_break, final_nb_pts, 
            r_score_feat, r_score_predict, breaking_cum_goal_diff, breaking_cum_goal_scored,
            roll_trend, trend]
