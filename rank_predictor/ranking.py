import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from xgboost import XGBRanker

# Implementation of xgb_ranker_raw (Rank-related)

"""
def get_xgboost_rank_ranker(training_data_df, validation_df, feature_cols):
    ranker = XGBRanker()
    nb_teams = validation_df.team.nunique()
    nb_training_seasons = training_data_df.season.nunique()
    group = np.array([nb_teams]*nb_training_seasons)
    
    training_data_df_sorted = training_data_df.sort_values(by='season').reset_index(drop=True)
    ranker.fit(X=training_data_df_sorted[feature_cols].values, 
           y=training_data_df_sorted['final_rank'].values,
           group=group)
    
    # ranker.predict(np.array([one_season_umap_valid[0]]))
    # the lower the better
    ranker_vals = ranker.predict(validation_df[feature_cols].values)
    tmp = pd.DataFrame(data=ranker_vals, columns=['xgb_ranker'])
    output_df = pd.concat([validation_df[['season', 'team', 'final_rank']], tmp], axis=1)
    output_df['predicted_rank'] = output_df['xgb_ranker'].rank()
    
    return output_df


def score_to_rank(season_df: pd.DataFrame, scores: np.array, col_name: str):
    tmp = pd.DataFrame(data=scores, columns=[col_name])

    output_df = pd.concat([season_df[['season', 'team', 'final_rank']], tmp], axis=1)

    output_df['predicted_rank'] = output_df[col_name].rank()
    return output_df
"""