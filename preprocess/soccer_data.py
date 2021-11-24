import pandas as pd
from copy import deepcopy

# functions to retrieve data


def rolling_mean_n_performance(df, window=5, performance_col='goals_scored'):
    dg = df.sort_values(by=['leg'])[['season', 'team', performance_col]].groupby(
        by=['season', 'team'])[performance_col].rolling(window=window, min_periods=1).mean().reset_index()
    
    new_col_name = f'rolling_{window}_games_avg_{performance_col}'

    df[new_col_name] = dg.set_index('level_2')[performance_col]
    return df


def get_past_feature(df, feat_col, team=True):
    merge_col = 'team' if team else 'opponent'    
    tmp_df = deepcopy(df[['season', 'leg', merge_col, feat_col]])
    tmp_df.loc[:, 'next_leg'] = tmp_df['leg'] + 1

    tmp_df.rename(columns={'leg': 'previous_leg', 
                           'next_leg': 'leg',
                           feat_col: f'previous_{merge_col}_{feat_col}'},
                  inplace=True)

    df = df.merge(tmp_df, how='left', on=['leg', 'season', merge_col])
    df.drop(columns=['previous_leg'], inplace=True)
    # print(f"length df : {len(df)}")
    return df


def prepare_data(csv_path, rolling=5):
    df = pd.read_csv(csv_path).drop(columns='Unnamed: 0')
    df['goal_diff'] = df['goals_scored'] - df['goals_conceded']
    # cumulative
    df['cum_pts'] = df[['season', 'team', 'nb_points']].groupby(by=['season', 'team']).cumsum()
    
    df['cum_goal_diff'] = df[['season', 'team', 'goal_diff']].groupby(by=['season', 'team']).cumsum()
    
    df['cum_goals_scored'] = df[['season', 'team', 'goals_scored']].groupby(by=['season', 'team']).cumsum()
    
    df['cum_goals_conceded'] = df['cum_goals_scored']-df['cum_goal_diff']
    df['rank'] = df[['season', 'leg', 'cum_pts', 'cum_goal_diff', 'cum_goals_scored']].sort_values(
        by=['cum_pts', 'cum_goal_diff', 'cum_goals_scored'], ascending=False).groupby(
        by=['season', 'leg']).cumcount() + 1
    
    df['avg_goals_scored_since_season_start'] = df['cum_goals_scored'].div(df['leg'])
    df['avg_goals_conceded_since_season_start'] = df['cum_goals_conceded'].div(df['leg'])
    df['avg_cum_pts_since_season_start'] = df['cum_pts'].div(df['leg'])
    
    # removed unwanted useless seasons
    data = deepcopy(df[df.season > '2003-2004'])
    data.reset_index(drop=True, inplace=True)
    
    leg_max = data.leg.max()
    
    end_season = data[data.leg == leg_max].rename(columns={'rank': 'final_rank', 'cum_pts': 'final_cum_pts'})
    data = data.merge(end_season[['season', 'team', 'final_rank', 'final_cum_pts']], on=['season', 'team'])
    
    # rolling mean
    cols = ['goals_conceded', 'goals_scored', 'nb_points']
    for c in cols:
        data = rolling_mean_n_performance(df=data, window=rolling, performance_col=c)
    # past features
    past_features = {'rank': [True, False], 
                     'rolling_5_games_avg_goals_scored': [True],
                     'rolling_5_games_avg_goals_conceded': [False],
                     'avg_goals_scored_since_season_start': [True],
                     'avg_goals_conceded_since_season_start': [False],
                     'goals_scored': [True],
                     'goals_conceded': [False],
                     'rolling_5_games_avg_nb_points': [True, False],
                     'nb_points': [True, False]
                     }

    for col, is_team_ll in past_features.items():
        for is_team in is_team_ll:
            data = get_past_feature(df=data, feat_col=col, team=is_team)
    
    return data
