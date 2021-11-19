import pandas as pd
import numpy as np
# from copy import deepcopy
# from datetime import datetime

from xgboost import XGBClassifier
# from metrics.soccer_ranking import get_rank_percentage_quality


# Implementation of xgb_class_raw (Classification)

class SoccerClassification:
    def __init__(self, nb_opponent) -> None:
        self.model = XGBClassifier()
        self.nb_opponent = nb_opponent
        self.championship_length = 2*(self.nb_opponent - 1)

    def train(self, feature_data, y, eval_metric='mlogloss') -> None:
        self.model.fit(X=feature_data, y=y, eval_metric=eval_metric)

    def get_ranking(self,
                    season_data: pd.DataFrame,
                    feature_cols: list,
                    teams: np.array,
                    predicted_rank_col: str = "classification_predicted_rank",
                    leg_col: str = 'leg'
                    ) -> pd.DataFrame:
        """ compute the probabilities to belong to the different classes """
        prob = self.model.predict_proba(X=season_data[feature_cols].values)
        weights = np.array([r + np.exp(np.log(100) * r / self.nb_opponent) for r in range(1, self.nb_opponent + 1)])
        scores = np.array([np.dot(prob[i], weights) for i in range(len(prob))])

        output_df = pd.DataFrame(data={'team': teams, 'score': scores})
        output_df[predicted_rank_col] = output_df['score'].rank()

        return output_df
