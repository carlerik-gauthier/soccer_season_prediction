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

    def get_ranking(self, data, teams: np.array, predicted_rank_col: str = "ranking_predicted_rank"):
        # TBD
        ranker_vals = self.model.predict(X=data)
        # the lower the better
        tmp = pd.DataFrame(data=ranker_vals, columns=['xgb_ranker'])
        output_df = pd.concat([data[['season', 'team', 'final_rank']], tmp], axis=1)
        output_df['predicted_rank'] = output_df['xgb_ranker'].rank()

        return output_df

    def get_training_performance(self, data, rank_col):
        ...