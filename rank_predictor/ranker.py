import os
import logging

from xgboost import XGBClassifier
from xgboost.sklearn import XGBRanker
from sklearn.linear_model import LinearRegression

from naive import SoccerNaive
from classification import SoccerClassification
from ranking import SoccerRanking
from regression import SoccerRegression

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=LOGLEVEL)
logger = logging.getLogger()


class Ranker:
    def __init__(self, feature_columns, type) -> None:
        self.feature_columns = feature_columns
        self.ranker_type = type if type in ['naive', 'regression', 'classification', 'ranking'] else 'naive'
        self.model = self._get_base_model()

    def _get_base_model(self):
        if self.ranker_type == 'classification':
            return SoccerClassification()
        elif self.ranker_type == 'regression':
            return SoccerRegression()
        elif self.ranker_type == 'ranking':
            return SoccerRanking()
        else:
            # will process Naive model
            return SoccerNaive()

    def train(self, train_data, target_column, eval_metric=None):

        eval_metric = None if self.ranker_type != 'classification' else eval_metric

        if eval_metric is not None:
            self.model.train(
                X=train_data[self.feature_columns].values,
                y=train_data[target_column].values,
                eval_metric=eval_metric  # 'mlogloss'
            )
        else:
            self.model.train(
                X=train_data[self.feature_columns].values,
                y=train_data[target_column].values
            )

    def get_ranking(self, data, teams):
        return self.model.get_ranking(data=data, teams=teams)

