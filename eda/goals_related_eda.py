import plotly.express as px

def hist_aggregator(df, column_to_describe, aggreg_column=None, bin_step=None):
    aggreg_column = column_to_describe if aggreg_column is None else aggreg_column
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x : (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
    
    feats = list({aggreg_column, column_to_describe})
    df_agg = df[feats+['country']].groupby(
        by=feats).count().reset_index()
    df_agg.rename(columns={'country': 'cnt'}, inplace=True)
    return df_agg

def mean_aggregator(df, column_to_describe, aggreg_column='play', bin_step=None):
    if bin_step is not None:
        df[f'{aggreg_column}_binned'] = df[aggreg_column].apply(lambda x : (x//bin_step)*bin_step)
        aggreg_column = f'{aggreg_column}_binned'
        
    df_agg = df[[aggreg_column, column_to_describe]].groupby(
        by=[aggreg_column]).mean().reset_index()
    df_agg.rename(columns={column_to_describe: f'avg_{column_to_describe}'}, inplace=True)
    return df_agg

# scatterplot on hist : change to hist bar
def draw_scatterplot(df, x, y, size_col, title):
    fig = px.scatter(df, x=x, y=y, size=size_col, title=title)
    fig.show()


# line on mean
def draw_line(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title)
    fig.show()