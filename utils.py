# TBD : complete train_model function

import os
import pickle
from pandas import DataFrame
from copy import deepcopy

MODEL_TYPE = ('regression', 'classification', 'ranking')


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
        raise ValueError("{name} is not available in {module}".format(name=file_name, module=module_path)
        )
    # load model
    for f in list_files(module_path=module_path):
        if f == file_name:
            return pickle.load(open(file_name+'.pickle', 'rb'))
    

def train_model(model_type: str,
                championship: str,
                # break_leg: int,
                train_data: DataFrame,
                validation_data: DataFrame):
    """
    :param model_type: str: type of rank predicter. Must be either 'regression', 'classification' or 'ranking'
    :param championship: str: name of the championship
    :param break_leg: int: number of legs that have already been played
    :param train_data: DataFrame: data to be used to train the model
    :param validation_data: DataFrame: data to be used to measure the performance of the model

    :returns: scikit-learn model
    """
    if model_type not in MODEL_TYPE:
        raise ValueError("model_type MUST be one the following values : {values}".format(values=', '.join(MODEL_TYPE)))
    
    # save model
    model_name = "{model_type}_{championship}_ranker".format(model_type=model_type, championship=championship)
    feature_colums = _get_feature_columns(data_df=train_data, model_type=model_type)
    if model_type == 'regression':
        # from rank_predictor.regression import
        model = ...
        pass
    elif model_type == 'classification':
        # from rank_predictor.classification import get_gradient_boosting_classifier_ranker
        model = ...
        pass
    else:
        # from rank_predictor.ranking import get_xgboost_rank_ranker
        model = ...

    # save model
    pickle.dump(model, open(model_name+'.pickle', 'wb'))

    return model


def list_files(module_path):
    return [f.split('.')[0] for f in os.listdir(module_path) if os.path.isfile(os.path.join(module_path, f))]


def _get_feature_columns(data_df: DataFrame, model_type='naive'):
    no_use_cols = ['index', 'country', 'season', 'team', 'opponent',
                   'previous_team_rolling_5_games_avg_goals_scored_binned']
    if model_type == 'classification':
        # on pivoted data
        return ['leg', 'play'] + [c for c in data_df.columns if c.startswith('previous') and c not in no_use_cols]
    elif model_type == 'regression':
        return ['lr_feat_coeff', 'nb_pts_at_break', 'cumulative_goal_diff_at_break',
                'rolling_5_avg_pts_at_break', 'nb_games_to_play_at_home']
    elif model_type == 'ranking':
        # on pivoted data
        return [...]
    else:
        # will process Naive model
        return [...]
