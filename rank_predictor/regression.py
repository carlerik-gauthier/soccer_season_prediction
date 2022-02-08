import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

# Implementation of lr_6 (Regression)


class SoccerRegression:
    def __init__(self, nb_opponent) -> None:
        self.model = LinearRegression()
        self.nb_opponent = nb_opponent
        self.championship_length = 2 * (self.nb_opponent - 1)

    def train(self, feature_data, y) -> None:
        self.model.fit(X=feature_data, y=y)

    def get_ranking(self,
                    season_data: pd.DataFrame,
                    feature_cols: list,
                    predicted_rank_col: str = "regression_predicted_rank"
                    ) -> pd.DataFrame:
        """ predict the linear slope from last known leg to end of the championship """
        try:
            season_data['predicted_linear_coeff'] = season_data[feature_cols].apply(
                lambda feat: self.model.predict(np.array(feat)), axis=1)
        except ValueError:
            season_data['predicted_linear_coeff'] = season_data[feature_cols].apply(
                lambda feat: self.model.predict(np.array(feat).reshape(1, -1))[0], axis=1)
        # predict the number of points by the end of the season
        cols = ['predicted_linear_coeff', 'nb_pts_at_break', 'break_leg']

        season_data['predicted_final_nb_pts'] = season_data[cols].apply(
            lambda r: r[1] + r[0] * (self.championship_length - r[2]), axis=1)
        # get final rank
        rank_df = season_data.sort_values(by='predicted_final_nb_pts', ascending=False).reset_index(drop=True)
        rank_df[predicted_rank_col] = rank_df.index + 1

        return rank_df

    def _predict(self, feature: np.array) -> float:
        feature = feature.reshape(1, -1)
        prediction = self.model.predict(feature)

        return prediction[0]


def get_lr_parameters(data: pd.DataFrame) -> tuple:
    tmp_x_data = data.leg.values
    tmp_y_data = data.cum_pts.values

    x_data = np.array([0] + list(tmp_x_data)).reshape(-1, 1)
    y_data = np.array([0] + list(tmp_y_data)).reshape(-1, 1)

    reg = LinearRegression(fit_intercept=False).fit(X=x_data, y=y_data)

    return reg.coef_[0][0], reg.score(X=x_data, y=y_data)
