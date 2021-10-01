from pandas import DataFrame

def get_team_participation(df: DataFrame, championship: str):
    nb_teams = df.team.nunique()
    nb_seasons = df.season.nunique()
    first_season = df.season.min()
    last_season = df.season.max()
    print(f"""{nb_teams} teams have played in {championship} from season {first_season} to season {last_season}, 
    i.e over {nb_seasons} seasons """)
    
    season_length = df.leg.max()
    
    end_season_df = df[df.leg==season_length].rename(columns={'rank':'final_rank'})
    participation_df = end_season_df[['team', 'final_rank']].groupby(by='team').agg('count').rename(
    columns={"final_rank":"nb_participation"})
    
    print("{nb_all_seasons} teams played all {nb_seasons} seasons".format(
        nb_all_seasons=len(participation_df[participation_df.nb_participation==nb_seasons]),
        nb_seasons=nb_seasons)
         )
    
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