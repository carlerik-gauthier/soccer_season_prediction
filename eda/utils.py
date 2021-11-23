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
    fig.show()


# line on mean
def draw_line(df: DataFrame, x: str, y: str, title: str):
    fig = px.line(df, x=x, y=y, title=title)
    fig.show()


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
