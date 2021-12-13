import os
import logging
import numpy as np
from pandas import DataFrame
from copy import deepcopy

from .naive import SoccerNaive
from .classification import SoccerClassification
from .ranking import SoccerRanking
from .regression import SoccerRegression
from metrics.soccer_ranking import get_rank_percentage_quality_dict

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
            return SoccerNaive(nb_opponent=self.nb_opponent)

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

    def get_performance(self,
                        test_data: DataFrame,
                        season_col: str,
                        real_rank_col: str,
                        real_final_points_col: str,
                        predicted_rank_col: str,
                        ranking_weight_version: int = 1):

        performance_ll = []
        for season in test_data[season_col].unique():
            test_data = deepcopy(test_data[test_data[season_col] == season])

            perf_score = self.compute_ranking_quality(test_data=test_data,
                                                      real_rank_col=real_rank_col,
                                                      real_final_points_col=real_final_points_col,
                                                      predicted_rank_col=predicted_rank_col,
                                                      ranking_weight_version=ranking_weight_version
                                                      )
            performance_ll.append(perf_score)

        return np.mean(performance_ll)

    def compute_ranking_quality(self,
                                test_data: DataFrame,
                                predicted_rank_col: str,
                                real_rank_col: str = 'final_rank',
                                real_final_points_col: str = 'final_cum_pts',
                                ranking_weight_version: int = 1
                                ):

        predicted_ranking_df = self.model.get_ranking(
            season_data=test_data,
            feature_cols=self.feature_columns,
            predicted_rank_col=predicted_rank_col)

        rank_weight = get_rank_percentage_quality_dict(nb_teams=self.nb_opponent, version=ranking_weight_version)
        base_val = deepcopy(test_data)[
            ['season', 'team', real_rank_col, real_final_points_col]].drop_duplicates().reset_index(drop=True)

        base_val['base_gain'] = base_val[[real_rank_col, real_final_points_col]].apply(
            lambda r: rank_weight[r[0]] + r[1], axis=1)

        deepcopy(base_val).sort_values(by='base_gain', ascending=False).reset_index(drop=True)

        base_val.sort_values(by='base_gain', ascending=False, inplace=True)
        base_val.reset_index(drop=True, inplace=True)
        base_val['inverse_position_discount'] = base_val.index + 1

        base_val['gain'] = base_val[['base_gain', 'position_discount']].apply(
            lambda r: r[0] / np.log2(1 + r[1]), axis=1)

        rank_to_inv_discount = {rk: pos_disc for rk, pos_disc in zip(base_val.final_rank,
                                                                     base_val.inverse_position_discount
                                                                     )
                                }

        predicted_ranking_df = predicted_ranking_df.merge(base_val[['team', 'base_gain']], on='team').rename(
            columns={'base_gain': f'base__{self.ranker_type}_gain'})

        predicted_ranking_df[f'{self.ranker_type}_gain'] = \
            predicted_ranking_df[[f'base__{self.ranker_type}_gain', predicted_rank_col]].apply(
                lambda r: r[0] / np.log2(1 + rank_to_inv_discount[r[1]]), axis=1)

        prediction_score = predicted_ranking_df[f'{self.ranker_type}_gain'].sum()
        truth_score = base_val.gain.sum()
        return round(100*prediction_score/truth_score, 2) if truth_score > 0 else 100

    def get_ranking(self, data, predicted_rank_col, teams=None):
        return self.model.get_ranking(season_data=data,
                                      feature_cols=self.feature_columns,
                                      predicted_rank_col=predicted_rank_col,
                                      teams=teams)
