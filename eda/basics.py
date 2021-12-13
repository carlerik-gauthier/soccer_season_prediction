from pandas import DataFrame


def get_team_participation(df: DataFrame):

    season_length = df.leg.max()
    
    end_season_df = df[df.leg == season_length]  # .rename(columns={'rank': 'final_rank'})
    participation_df = end_season_df[['team', 'final_rank']].groupby(by='team').agg('count').rename(
        columns={"final_rank": "nb_participation"})
    print(end_season_df[['team', 'final_rank']].groupby(by='team').agg('count').head())
    return participation_df.sort_values(by="nb_participation", ascending=False)


def get_goal_scored_repartition(data_df: DataFrame):
    dg = data_df[['championship', 'goals_scored', 'play']].groupby(
        by=['championship', 'goals_scored']).count()
    dg.reset_index(inplace=True)
    dg.rename(columns={'play': 'quantity'}, inplace=True)
    total = data_df[['championship', 'play']].groupby(by=['championship']).count()
    total.reset_index(inplace=True)
    total.rename(columns={'play': 'total'}, inplace=True)
    class_recap = dg.merge(total, how='left', on='championship')
    class_recap['percent'] = 100*class_recap['quantity'].div(class_recap['total'])
    return class_recap


def get_nb_competitor(df: DataFrame, leg_col: str = 'leg'):
    max_leg = df[leg_col].max()
    return 1 + (max_leg/2)


def get_championship_length(df: DataFrame, leg_col: str = 'leg'):
    return df[leg_col].max()
