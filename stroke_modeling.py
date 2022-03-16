# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import plotly.offline as pyo
import plotly.graph_objs as go

# %%


def numeralize_categories(df: pd.DataFrame) -> list:
    '''
    receives a dataframe identifies the columns of type category and returns a list of tupples, each tupple 
    contains the name of the column and a dictionary with each unique value as a key and the number associated 
    with it.
    '''
    cats = []
    for column in df.columns:
        if df.dtypes[column] == 'category':
            cats.append(
                (column, {j: i for i, j in enumerate(df[column].unique(), 0)}))
    return cats

# %%


def numeralize_column(col: pd.Series) -> tuple:
    '''
    receives a series evaluates if is an object type and returns a tupple with the name of the series and
    a dictionary with the unique values asociated
    '''
    if col.dtype == 'object':
        return col.name, {j: i for i, j in enumerate(col.unique(), 0)}

# %%


def numeralize_columns(categories_map: list, df: pd.DataFrame):
    '''
    receives a list of tupples (category name, dictionary{unique values}) and a dataframe. The dataframe columns
    contained in the categories_map list are replaced with the values of the dictionary 
    '''
    for cat, items in categories_map:
        df[cat].replace(items, inplace=True)


# %%
def scatter_plot(x: pd.Series, y: pd.Series, color: str):
    data = [go.Scatter(x=x,
                       y=y,
                       mode='markers',
                       marker=dict(size=12,
                                   color=color,
                                   symbol='circle-dot',
                                   line={'width': 2}))]
    layout = go.Layout(title='{} vs {} correlation'.format(x.name, y.name),
                       xaxis={'title': x.name},
                       yaxis={'title': y.name},
                       hovermode='closest',
                       template='plotly_white')
    fig = go.Figure(data=data, layout=layout)

    fig.show()
    # pyo.plot(fig, filename='{}_{}_correlation.html'.format(x.name, y.name))

# %%


def get_categories_bar_plot(df: pd.DataFrame, categories: list, value_name: str):
    data = [go.Bar(x=df.index, y=df[cat], name=cat) for cat in categories]
    layout = go.Layout(title='{} by {}'.format(value_name, df.columns.name))

    fig = go.Figure(data=data, layout=layout)
    fig.show()

# %%


def get_scatter_with_marker(df: pd.DataFrame, metadata: dict):
    '''
      metadata {
        x: x axis
        y: y axis
        text: each point label
        mode: chart mode
        marker: markers parameters
      }
    '''
    data = [go.Scatter(x=df[metadata['x']],
                       y=df[metadata['y']],
                       text=df[metadata['text']],
                       mode=metadata['mode'],
                       marker=dict(size=df[metadata['marker']]))]
    layout = go.Layout(title='{} by {} and {}'.format(
        metadata['x'], metadata['y'], metadata['marker']))
    return go.Figure(data=data, layout=layout)


# %%
"""
Separate X and Y
"""


def get_variables(df: pd.DataFrame, dependent: str) -> list:
    '''
    Receives the desired dataframe to separate and the column name to be identified as dependent variable
    returns a list containing a DataFrame for independent variables and a Series with the dependent variable
    '''
    if dependent in df.columns:
        x = df.copy(deep=True)
        y = x[dependent]
        x.drop('stroke', axis='columns', inplace=True)
        return [x, y]
    else:
        return None


# %%
file = 'data/healthcare-dataset-stroke-data.csv'

# %%
raw_data = pd.read_csv(file)

# %%
raw_data.drop(['id'], axis=1, inplace=True)

# %%
raw_data[raw_data.isna().values]

# %%
for column in raw_data.columns:
    print(raw_data[column].unique())

# %%
for i, row in raw_data.iterrows():
    if pd.isna(row['bmi']):
        raw_data.iat[i, raw_data.columns.get_loc(
            'bmi')] = raw_data.loc[raw_data['age'] == row['age']].bmi.median()

# %%
scatter_plot(raw_data.age, raw_data.bmi, '#dab894')

# %%
scatter_plot(raw_data.avg_glucose_level, raw_data.bmi, '#dab894')

# %%
stroke_by_smoking = pd.pivot_table(raw_data, values='stroke', index=[
                                   'smoking_status'], columns=['gender'], aggfunc=np.sum)

# %%
get_categories_bar_plot(
    stroke_by_smoking, stroke_by_smoking.columns, 'strokes')

# %%
df_stroke_never_smoked_F = raw_data.loc[(raw_data['stroke'] == 1) & (
    raw_data.gender == 'Female') & (raw_data['smoking_status'] == 'never smoked')]

# %%
df_stroke_never_smoked_F

# %%
params = {'x': 'avg_glucose_level',
          'y': 'age',
          'text': 'bmi',
          'mode': 'markers',
          'marker': 'bmi'
          }

get_scatter_with_marker(df_stroke_never_smoked_F, params).show()

# %%
df_female_stroke = raw_data.loc[(raw_data['stroke'] == 1) & (
    raw_data['gender'] == 'Female')]

# %%
df_female_stroke

# %%
get_scatter_with_marker(df_female_stroke, params).show()

# %%
df_male_stroke = raw_data.loc[(raw_data['stroke'] == 1) & (
    raw_data['gender'] == 'Male')]

# %%
get_scatter_with_marker(df_male_stroke, params).show()
