# TBD : complete train_model function

import os
import pickle
from pandas import DataFrame
# from copy import deepcopy
from rank_predictor.ranker import Ranker

MODEL_TYPE = ('regression', 'classification', 'ranking')
RANKING_WEIGHT_VERSION = 1
SEASON_COL = 'season'
REAL_RANK_COL = 'final_rank'
REAL_FINAL_POINTS_COL = 'final_nb_pts'
PREDICTED_RANK_COL = 'predicted_rank'


def is_available(module_path: str, file_name: str):
    """
    :param module_path: str: path to the module
    :param file_name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    # mypath = os.path.join(os.getcwd(), 'misc')
    return file_name in list_files(module_path=module_path)


def retrieve_model(module_path: str, file_name: str):
    """
    :param module_path: str: path to the the module
    :param file_name: str: name of the file to find

    :returns bool: True if the file exists in module else False
    """
    if not is_available(module_path=module_path, file_name=file_name):
        raise ValueError("{name} is not available in {module}".format(name=file_name, module=module_path))
    # load model
    for f in list_files(module_path=module_path):
        if f == file_name:
            return pickle.load(open(file_name+'.pickle', 'rb'))
    

def train_model(model_type: str,
                championship: str,
                nb_opponent: int,
                train_data: DataFrame
                ):
    """
    :param model_type: str: type of rank predicter. Must be either 'regression', 'classification' or 'ranking'
    :param championship: str: name of the championship
    :param nb_opponent: int: number of teams taking part to the competition
    :param train_data: DataFrame: data to be used to train the model

    :returns: scikit-learn model
    """
    if model_type not in MODEL_TYPE:
        raise ValueError("model_type MUST be one the following values : {values}".format(values=', '.join(MODEL_TYPE)))
    
    # save model
    model_name = "{model_type}_{championship}_ranker".format(model_type=model_type, championship=championship)
    feature_colums = _get_feature_columns(data_df=train_data, model_type=model_type)
    model = Ranker(feature_columns=feature_colums, ranker_type=model_type, nb_opponent=nb_opponent)
    if model_type in ['ranking', 'classification', 'regression']:
        model.train(train_data=train_data, target_column=_get_target_col(model_type=model_type))
    # if model_type == 'regression':
    #     model.train(train_data=train_data, target_column=...)
    # elif model_type == 'classification':
    #     model = ...
    # elif model_type == 'ranking':
    #     model = ...
    else:
        model = None

    # save model
    if model_type in ['ranking', 'classification', 'regression']:
        pickle.dump(model, open(model_name+'.pickle', 'wb'))

    return model


def get_model_performance(test_data: DataFrame, model: Ranker):
    return model.get_performance(test_data=test_data,
                                 season_col=SEASON_COL,
                                 real_rank_col=REAL_RANK_COL,
                                 real_final_points_col=REAL_FINAL_POINTS_COL,
                                 predicted_rank_col=PREDICTED_RANK_COL,
                                 ranking_weight_version=RANKING_WEIGHT_VERSION
                                 )


def list_files(module_path):
    return [f.split('.')[0] for f in os.listdir(module_path) if os.path.isfile(os.path.join(module_path, f))]


def _get_feature_columns(data_df: DataFrame, model_type='naive'):
    no_use_cols = ['index', 'country', 'season', 'team', 'opponent',
                   'previous_team_rolling_5_games_avg_goals_scored_binned']
    if model_type == 'classification':
        # on pivoted data
        return ['leg', 'play'] + [c for c in data_df.columns if c.startswith('previous') and c not in no_use_cols]
    elif model_type == 'ranking':
        # on pivoted data
        feat_cols = [c for c in data_df.columns if c.startswith('leg')]
        feat_cols += ['rolling_5_games_avg_nb_points', 'avg_cum_pts_since_season_start',
                      'cum_goal_diff', 'cum_goals_scored']
        return feat_cols
    else:
        # for naive and regression model_type
        return ['lr_feat_coeff', 'nb_pts_at_break', 'cumulative_goal_diff_at_break',
                'rolling_5_avg_pts_at_break', 'nb_games_to_play_at_home']


def _get_target_col(model_type='naive'):
    if model_type == 'regression':
        return 'lr_predict_coeff'
    elif model_type == 'classification':
        return 'final_rank'
    elif model_type == 'ranking':
        return 'final_rank'
    else:
        return None
