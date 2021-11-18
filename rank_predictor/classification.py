import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from xgboost import XGBClassifier
from metrics.soccer_ranking import get_rank_percentage_quality


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
                    teams: np.array,
                    predicted_rank_col: str = "classification_predicted_rank",
                    leg_col: str = 'leg'
                    ) -> pd.DataFrame:
        """ compute the probabilities to belong to the different classes """
        probs = self.model.predict_proba(X=season_data)
        weights = np.array([r + np.exp(np.log(100) * r / self.nb_opponent) for r in range(1, self.nb_opponent + 1)])
        scores = np.array([np.dot(probs[i], weights) for i in range(len(probs))])

        output_df = pd.DataFrame(data={'team': teams, 'score': scores})
        output_df[predicted_rank_col] = output_df['score'].rank()

        return output_df


"""
def get_gradient_boosting_classifier_ranker(training_data_df, 
                                            validation_df,
                                            feature_cols
                                           ):

    nb_teams = validation_df.team.nunique()

    classifier = XGBClassifier()
    core = 'xgb_classifier'

    classifier.fit(X=training_data_df[feature_cols].values, y=training_data_df['final_rank'].values, 
                 eval_metric='mlogloss')

    # compute the probabilities to belong to the different classes
    probs = classifier.predict_proba(validation_df[feature_cols].values)

    weights = np.array([r+np.exp(np.log(100)*r/nb_teams) for r in range(1, nb_teams + 1)])

    evaluation = np.array([np.dot(probs[i], weights) for i in range(len(probs))])

    return score_to_rank(season_df=validation_df, 
                          scores=evaluation, 
                          col_name=f'{core}')

"""

"""
get_gradient_boosting_classifier_ranker(
        training_data_df=deepcopy(train_pivoted_df),
        validation_df=deepcopy(valid_pivoted_df),
        feature_cols=feat_cols,
        model_type='simple_classifier')
"""
