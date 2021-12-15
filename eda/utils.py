import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame


# scatterplot on hist : change to hist bar
def draw_scatterplot(df: DataFrame,
                     x: str,
                     y: str,
                     size_col: str,
                     title: str):
    fig = px.scatter(df, x=x, y=y, size=size_col, title=title)
    return fig


# line on mean
def draw_line(df: DataFrame, x: str, y: str, title: str):
    fig = px.line(df, x=x, y=y, title=title)
    return fig


def draw_pie_chart(df, values, names, title):
    if isinstance(names, list):
        name = '_'.join(names)
        df[name] = df[names].apply(lambda r: '_'.join([str(_) for _ in r]), axis=1)
    else:
        name = names

    fig = go.Figure(data=[go.Pie(labels=df[name], values=df[values])])
    fig.update_traces(title_text=title, textposition='inside', textinfo='percent+label')
    return fig


def draw_sunburst(df, path, values, color=None):
    fig = px.sunburst(df, path=path, values=values, color=color)
    return fig


def get_layout():
    return go.Layout(autosize=True,
                     width=800,
                     height=800,
                     xaxis=go.layout.XAxis(linecolor='black',
                                           linewidth=1,
                                           mirror=True),

                     yaxis=go.layout.YAxis(linecolor='black',
                                           linewidth=1,
                                           mirror=True),

                     margin=go.layout.Margin(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4
                     )
                     )


def get_layers_avg_kpi(plot_name: str,
                       x: pd.Series,
                       avg_data: pd.Series,
                       color: str = 'blue',
                       width: int = 2,
                       std_data: pd.Series = None,
                       fillcolor: str = 'green'):
    sublayer = [go.Scatter(name=plot_name,
                           x=x,
                           y=avg_data,
                           mode='lines',
                           line=dict(color=color,
                                     width=width
                                     )
                           )
                ]
    if std_data is None:
        return sublayer

    sublayer += [go.Scatter(name=f'Upper Bound {plot_name}',
                            x=x,
                            y=avg_data + std_data,
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            showlegend=False
                        ),

                 go.Scatter(name=f'Lower Bound {plot_name}',
                            x=x,
                            y=avg_data - std_data,
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            mode='lines',
                            fillcolor=fillcolor,
                            fill='tonexty',
                            showlegend=False,
                            )
                 ]
    return sublayer


def get_layer_cumulative_kpi(plot_name: str,
                             x: pd.Series,
                             y: pd.Series,
                             color: str = "blue",
                             width: int = 2):
    return [go.Scatter(name=plot_name,
                       x=x,
                       y=y,
                       mode='lines',
                       line=dict(color=color,
                                 width=width)
                       )
            ]
