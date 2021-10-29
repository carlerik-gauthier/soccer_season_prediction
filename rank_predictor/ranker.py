import os
import logging

from xgboost import XGBClassifier

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
        if self.ranker_type=='classification':
            return XGBClassifier()
        elif self.ranker_type=='regression':
            ...
        elif self.ranker_type=='ranking':
            ...
        else:
            ...
    
    def train(self, train_data, target_column, eval_metric=None):
        if eval_metric:
            self.model.fit(
                X=train_data[self.feature_columns].values,
                y=train_data[target_column].values, 
                eval_metric=eval_metric # 'mlogloss'
                )
        else:
            self.model.fit(
                X=train_data[self.feature_columns].values,
                y=train_data[target_column].values
            )

    def predict(self, data):
        ...

