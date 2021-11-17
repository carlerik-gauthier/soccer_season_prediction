import numpy as np
from pandas import DataFrame
from metrics.soccer_ranking import get_rank_percentage_quality

class SoccerNaive:
    def __init__(self) -> None:
        pass

    def train(self):
        # no training is performed. However it is necessary for general flow
        pass

    def get_ranking(self, season_data: DataFrame, championship_length: int, predicted_rank_col: str = "naive_predicted_rank"):
        cols = ['lr_feat_coeff']
        # get the predicted number of points by the end of the season
        season_data['predicted_final_nb_pts'] = season_data[cols].apply(lambda r: r*championship_length, axis=1)

        rank_df = season_data.sort_values(by='predicted_final_nb_pts', ascending=False).reset_index(drop=True)
        rank_df[predicted_rank_col] = rank_df.index +1
        return rank_df


    def get_training_performance(self, test_data: DataFrame, real_rank_col: str, predicted_rank_col: str = "naive_predicted_rank"):
        ...
