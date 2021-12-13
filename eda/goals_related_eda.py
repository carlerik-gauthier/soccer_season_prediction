from pandas import DataFrame


def hist_aggregator(df: DataFrame,
                    column_to_describe: str,
                    aggreg_column: str = None,
                    bin_step: float = None):
    """
    The purpose of this function is to get a histogram for a variable from the given dataframe based
    on groups defined by another variable
    """
    aggreg_column = column_to_describe if aggreg_column is None else aggreg_column
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x: (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
    
    feats = list({aggreg_column, column_to_describe})
    df_agg = df[feats+['country']].groupby(
        by=feats).count().reset_index()
    df_agg.rename(columns={'country': 'cnt'}, inplace=True)
    return df_agg


def mean_aggregator(df: DataFrame,
                    column_to_describe: str,
                    aggreg_column: str = 'play',
                    bin_step: float = None):
    """
    The purpose of this function is to get average values for a variable from the given dataframe based on groups
     defined by another variable
    """
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x: (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
        
    df_agg = df[[aggreg_column, column_to_describe]].groupby(
        by=[aggreg_column]).mean().reset_index()
    df_agg.rename(columns={column_to_describe: f'avg_{column_to_describe}'}, inplace=True)
    return df_agg

# Some plot functions ?
