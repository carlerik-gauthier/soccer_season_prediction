import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MS

from xgboost import XGBClassifier, XGBRanker

# 3 models : lr_6 (Regression), xgb_class_raw (Classification), xgb_ranker_raw (Rank-related)