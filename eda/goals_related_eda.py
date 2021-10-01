import plotly.express as px
from pandas import DataFrame

def hist_aggregator(df: DataFrame, column_to_describe: str, aggreg_column: str=None, bin_step: int=None):
    """
    The purpose of this function is to get an histogram for a variable from the given dataframe based on groups defined by another variable
    """
    aggreg_column = column_to_describe if aggreg_column is None else aggreg_column
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x : (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
    
    feats = list({aggreg_column, column_to_describe})
    df_agg = df[feats+['country']].groupby(
        by=feats).count().reset_index()
    df_agg.rename(columns={'country': 'cnt'}, inplace=True)
    return df_agg

def mean_aggregator(df: DataFrame, column_to_describe: str, aggreg_column: str='play', bin_step: int=None):
    """
    The purpose of this function is to get average valeus for a variable from the given dataframe based on groups defined by another variable
    """
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x : (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
        
    df_agg = df[[aggreg_column, column_to_describe]].groupby(
        by=[aggreg_column]).mean().reset_index()
    df_agg.rename(columns={column_to_describe: f'avg_{column_to_describe}'}, inplace=True)
    return df_agg

# scatterplot on hist : change to hist bar
def draw_scatterplot(df: DataFrame, x: str, y: str, size_col: str, title: str):
    fig = px.scatter(df, x=x, y=y, size=size_col, title=title)
    fig.show()


# line on mean
def draw_line(df: DataFrame, x: str, y: str, title: str):
    fig = px.line(df, x=x, y=y, title=title)
    fig.show()