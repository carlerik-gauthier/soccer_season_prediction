import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from xgboost import XGBRanker

# Implementation of xgb_ranker_raw (Rank-related)

class SoccerRanking:
    def __init__(self) -> None:
        self.model = XGBRanker()
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)

    def get_ranking(self, data, teams: np.array):
        ranker_vals = self.model.predict(X=data)
        output_df = pd.DataFrame(data={'team': teams, 'score': ranker_vals})
        output_df['classification_predicted_rank'] = output_df['score'].rank()
        output_df['predicted_rank'] = output_df['xgb_ranker'].rank()
        
        return output_df

    def get_training_performance(self, data, rank_col):
        ...




"""def get_xgboost_rank_ranker(training_data_df, validation_df, feature_cols):
    ranker = XGBRanker()
    nb_teams = validation_df.team.nunique()
    nb_training_seasons = training_data_df.season.nunique()
    group = np.array([nb_teams]*nb_training_seasons)
    
    training_data_df_sorted = training_data_df.sort_values(by='season').reset_index(drop=True)
    ranker.fit(X=training_data_df_sorted[feature_cols].values, 
           y=training_data_df_sorted['final_rank'].values,
           group=group)
    
    return ranker
"""
"""
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


get_xgboost_rank_ranker(
    training_data_df=deepcopy(train_pivoted_df),
    validation_df=deepcopy(valid_pivoted_df),
    feature_cols=feat_cols)
"""