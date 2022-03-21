import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash import Input, Output, State
import pandas as pd

# Dash Bootstrap Documentation https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/

file_path = 'data/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)

df.drop(['id'], axis=1, inplace=True)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])

categorical_cols = list(df.select_dtypes(include='object'))
numeric_cols = []

collapse = html.Div(
    [
        dbc.Button(
            'Preview Dataset',
            id='collapse-button',
            class_name='mb-3',
            color='primary',
            size='sm',
            n_clicks=0
        ),
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    dbc.Table.from_dataframe(
                        df.head(), striped=True, hover=True, color='dark', responsive=True)
                )
            ),
            id='collapse',
            is_open=False
        )
    ]
)


inline_checklist = html.Div(
    [
        dbc.Label("Choose the categorical variables"),
        dbc.Checklist(
            options=[
                {'label': item, 'value': df.columns.get_loc(item)} for item in df.columns
            ],
            value=[],
            id="checklist-inline-input",
            inline=True,
        ),
    ]
)

app.layout = html.Div([collapse, inline_checklist])


@app.callback(
    Output('collapse', 'is_open'),
    [Input('collapse-button', 'n_clicks')],
    [State('collapse', 'is_open')]
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server()
