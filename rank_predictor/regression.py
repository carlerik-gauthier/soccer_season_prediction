import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MS

# Implementation of lr_6 (Regression)

class SoccerRegression:
    def __init__(self) -> None:
        self.model = LinearRegression()
    
    def train(self, X, y):
        self.model.fit(X=X, y=y)

    def get_ranking(self, data, teams: np.array):
        pass

    def get_training_performance(self, data, rank_col):
        ...


def get_lr_parameters(data: pd.DataFrame):
    tmp_x_data = data.leg.values
    tmp_y_data = data.cum_pts.values
    
    x_data = np.array([0] + list(tmp_x_data)).reshape(-1, 1)
    y_data = np.array([0] + list(tmp_y_data)).reshape(-1, 1)
    
    reg = LinearRegression(fit_intercept=False).fit(X=x_data, y=y_data)
    
    return reg.coef_[0][0] , reg.score(X=x_data, y=y_data)


"""
lr_6, meta = ranker(data_training=deepcopy(training_data_for_model),
                        data_evaluation=deepcopy(validation_data_for_model),
                        feature_cols=feature_cols,
                        model_type='lin_reg',
                        breaking_leg=breaking_leg,
                        final_leg=final_leg)



 naive = basic(data_training=deepcopy(training_data_for_model),
                   data_evaluation=deepcopy(validation_data_for_model),
                   feature_cols=feature_cols,
                   breaking_leg=breaking_leg,
                   final_leg=final_leg)



def fit_general_model(
    data: pd.DataFrame, 
    feature_cols, 
    target_col, 
    test_frac = .2,
    model_type = 'lin_reg'):
    
    assert model_type in ['lin_reg', 'xgboost', 'random_forest']
    
    data.reset_index(drop=True, inplace=True)
    
    
    features = data[feature_cols].values
    target = data[target_col].values
    
    # split data
    x_train, x_test, y_train, y_test = train_test_split(features, 
                                                        target,
                                                        test_size=test_frac,
                                                        random_state=42)
    
    # get model
    if model_type=='random_forest':
        model = RandomForestRegressor().fit(X=x_train, y=y_train)
    elif model_type=='xgboost':
        model = XGBRegressor().fit(X=x_train, y=y_train)
    else:
        model = LinearRegression().fit(X=x_train, y=y_train)
        
    pred = model.predict(x_test) 

    # R2_score :
    r2_score_train = model.score(X=x_train, y=y_train)
    r2_score_test = model.score(X=x_test, y=y_test)

     # RMSE Computation 
    rmse = np.sqrt(MSE(y_test, pred))
    
    return model, {'r2_score_train': r2_score_train, 'r2_score_test': r2_score_test, 'rmse_test': rmse} 


def predict(scikit_model, feature: np.array):

    feature = feature.reshape(1, -1)
    prediction = scikit_model.predict(feature)
    
    return prediction[0]


def compute_expected_final_nb_points(lin_coeff, nb_pts_at_break, breaking_leg, final_leg, basic=True):
    # print(f'inputs are : lin_coeff={lin_coeff}, nb_pts_at_break={nb_pts_at_break}, breaking_leg={breaking_leg} and final_leg={final_leg}')
    # print(f"-- basic is {basic}")
    if basic:
      #  print(f'---- output is {lin_coeff*final_leg}')
        return lin_coeff*final_leg

    else:
        # print(f'---- output is {nb_pts_at_break + lin_coeff*(final_leg - breaking_leg)}')
        return nb_pts_at_break + lin_coeff*(final_leg - breaking_leg)

def basic(data_training, data_evaluation, feature_cols, breaking_leg=27, final_leg=38):
    cols = ['lr_feat_coeff', 'nb_pts_at_break']

    data_evaluation['predicted_final_nb_pts'] = data_evaluation[cols].apply(
        lambda r: compute_expected_final_nb_points(lin_coeff=r[0],
                                                   nb_pts_at_break=r[1],
                                                   breaking_leg=breaking_leg,
                                                   final_leg=final_leg, 
                                                   basic=True), 
        axis=1)

    return data_evaluation

def ranker(data_training, data_evaluation, feature_cols, model_type,
              breaking_leg=27, final_leg=38):

    # fit the model
    model, metadata = fit_general_model(
        data=data_training, 
        feature_cols=feature_cols, 
        target_col='lr_predict_coeff',
        model_type=model_type)
    # get the predicted number of points
    data_evaluation[f'predicted_{model_type}_predict_coeff'] = data_evaluation[feature_cols].apply(
        lambda x: predict(scikit_model=model, 
                          feature=np.array(x)),
        axis=1)

    cols = [f'predicted_{model_type}_predict_coeff', 'nb_pts_at_break']
        
    data_evaluation['predicted_final_nb_pts'] = data_evaluation[cols].apply(
        lambda r: compute_expected_final_nb_points(lin_coeff=r[0],
                                                   nb_pts_at_break=r[1],
                                                   breaking_leg=breaking_leg,
                                                   final_leg=final_leg, 
                                                   basic=False), 
        axis=1)
    
    rmse = np.sqrt(MSE(data_evaluation['final_nb_pts'].values,
                       data_evaluation[f'predicted_final_nb_pts'].values)
                  )
    metadata['rmse_eval'] = rmse
    
    return data_evaluation, metadata

def points_predicter(data_training, data_evaluation, feature_cols, target_col, model_type):

    model, metadata = fit_general_model(data=data_training, 
                                        feature_cols=feature_cols, 
                                        target_col=target_col,
                                        model_type=model_type)
    # get the predicted number of points
    data_evaluation[f'predicted_{target_col}'] = data_evaluation[feature_cols].apply(
        lambda x: predict(scikit_model=model, 
                          feature=np.array(x)),
        axis=1)

    # RMSE Computation 
    rmse = np.sqrt(MSE(data_evaluation[target_col].values,
                       data_evaluation[f'predicted_{target_col}'].values)
                  )
    metadata['rmse_eval'] = rmse
    return data_evaluation, metadata

def points_to_rank(season_data: pd.DataFrame, pts_col_name: str, rank_name: str):
    rank_df = season_data.sort_values(by=pts_col_name, ascending=False).reset_index(drop=True)
    rank_df[rank_name] = rank_df.index +1
    return rank_df

ranker(data_training=deepcopy(training_data_for_model),
                        data_evaluation=deepcopy(validation_data_for_model),
                        feature_cols=feature_cols,
                        model_type='lin_reg',
                        breaking_leg=breaking_leg,
                        final_leg=final_leg)
"""