import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from xgboost import XGBClassifier

# Implementation of xgb_class_raw (Classification)

def get_gradient_boosting_classifier_ranker(training_data_df, 
                                            validation_df,
                                            feature_cols,
                                            model_type='simple_classifier'
                                           ):
                                    
    pass                                      
    """
    # training_data_df = train_pivoted_df[feat_cols] 
    assert model_type in ['simple_classifier', 'rf_classifier']
    
    nb_teams = validation_df.team.nunique()
    
    if model_type == 'rf_classifier':
        classifier = XGBRFClassifier()
        core = 'xgb'
    else:
        classifier = XGBClassifier()
        core = 'xgbrf'
    
    classifier.fit(X=training_data_df[feature_cols].values, y=training_data_df['final_rank'].values, 
                 eval_metric='mlogloss')
    
    # compute the probabilities to belong to the different classes
    probs = classifier.predict_proba(validation_df[feature_cols].values)
    
    weights = np.array([r+np.exp(np.log(100)*r/nb_teams) for r in range(1, nb_teams + 1)])
    
    evaluation = np.array([np.dot(probs[i], weights) for i in range(len(probs))])
    
    return score_to_rank(season_df=validation_df, 
                          scores=evaluation, 
                          col_name=f'{core}_classifier_umap')
"""