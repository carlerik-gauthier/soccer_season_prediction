import os
import logging
import numpy as np
from pandas import DataFrame
from copy import deepcopy

from naive import SoccerNaive
from classification import SoccerClassification
from ranking import SoccerRanking
from regression import SoccerRegression

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()


class Ranker:
    def __init__(self, feature_columns, ranker_type, nb_opponent) -> None:
        self.feature_columns = feature_columns
        self.ranker_type = ranker_type if ranker_type in ['naive', 'regression', 'classification', 'ranking'] \
            else 'naive'
        self.model = self._get_base_model()
        self.nb_opponent = nb_opponent

    def _get_base_model(self):
        if self.ranker_type == 'classification':
            return SoccerClassification(nb_opponent=self.nb_opponent)
        elif self.ranker_type == 'regression':
            return SoccerRegression(nb_opponent=self.nb_opponent)
        elif self.ranker_type == 'ranking':
            return SoccerRanking(nb_opponent=self.nb_opponent)
        else:
            # will process Naive model
            return SoccerNaive()

    def train(self, train_data: DataFrame, target_column, eval_metric=None):

        eval_metric = None if self.ranker_type != 'classification' else eval_metric

        if eval_metric is not None:
            self.model.train(
                feature_data=train_data[self.feature_columns].values,
                y=train_data[target_column].values,
                eval_metric=eval_metric  # 'mlogloss'
            )
        elif self.ranker_type == 'ranking':
            nb_training_season = train_data.shape[0]/self.nb_opponent
            group = np.array([self.nb_opponent] * nb_training_season)
            self.model.train(feature_data=train_data[self.feature_columns].values,
                             y=train_data[target_column].values,
                             group=group)
        else:
            self.model.train(
                feature_data=train_data[self.feature_columns].values,
                y=train_data[target_column].values
            )

    def get_training_performance(self, test_data: DataFrame,
                                 season_col: str,
                                 real_rank_col: str, predicted_rank_col: str):

        for season in test_data[season_col].unique():
            predicted_ranking_df = self.model.get_ranking(
                season_data=deepcopy(test_data[test_data[season_col] == season]),
                feature_cols=self.feature_columns,
                predicted_rank_col=predicted_rank_col)
            ...

        # self.model.get_training_performance(test_data=test_data,
        #                                     real_rank_col=real_rank_col,
        #                                     predicted_rank_col=predicted_rank_col)

    def get_ranking(self, data, predicted_rank_col, teams=None):
        return self.model.get_ranking(season_data=data,
                                      feature_cols=self.feature_columns,
                                      predicted_rank_col=predicted_rank_col,
                                      teams=teams)

