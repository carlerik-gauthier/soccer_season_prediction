import pandas as pd
import numpy as np

from xgboost import XGBRanker


# Implementation of xgb_ranker_raw (Rank-related)

class SoccerRanking:
    def __init__(self, nb_opponent) -> None:
        self.model = XGBRanker()
        self.nb_opponent = nb_opponent
        self.championship_length = 2 * (self.nb_opponent - 1)

    def train(self, feature_data, y, group):
        # group = np.array([20]*(15-len(validate_season)))
        # train_pivoted_df_sorted = train_pivoted_df.sort_values(by='season').reset_index(drop=True)
        # ranker_2.fit(X=train_pivoted_df_sorted[feat_cols].values,
        #            y=train_pivoted_df_sorted['final_rank'].values,
        #            group=group)
        self.model.fit(X=feature_data, y=y, group=group)

    def get_ranking(self, season_data: pd.DataFrame,
                    feature_cols: list,
                    predicted_rank_col: str = "ranking_predicted_rank",
                    teams: np.array = None,
                    leg_col: str = 'leg'):
        # TBD
        ranker_vals = self.model.predict(X=season_data[feature_cols].values)
        # the lower the better
        tmp = pd.DataFrame(data=ranker_vals, columns=['xgb_ranker'])
        output_df = pd.concat([season_data[['season', 'team', 'final_rank']], tmp], axis=1)
        output_df[predicted_rank_col] = output_df['xgb_ranker'].rank()

        return output_df
