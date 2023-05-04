import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

df = pd.read_csv(f"{os.getcwd()}/data/fdic_paragraphs_improved.csv")
vectors = pd.read_csv(f"{os.getcwd()}/data/stored_vectors_roberta_improved.csv")

roberta = SentenceTransformer(f"{os.getcwd()}/models/finetuned_roberta")


def ask_question(question, vector_df, text_df, model):
    q_vec = model.encode([question])
    cosine_mat = cosine_similarity(q_vec, vector_df)
    top_5_index = np.argsort(cosine_mat[0])[::-1][:5]
    output_strings = []
    for i, v in enumerate(top_5_index):
        record = text_df.iloc[v]
        output_strings.append(
        {
            "index": i+1,
            "Section": record.SECTNO,
            "Subject": record.SUBJECT,
            "Text": record.TEXT + '\n',
            "Similarity_Score": f'{round(cosine_mat[0][v]*100, 2)}%',
        }
        )
    return output_strings

app = dash.Dash(__name__)

col_vals = ['Subject', 'Text']

data_style = [{
                    'if': {'column_id': col},
                    'whiteSpace': 'normal',
                    'height': 'auto',
                } for col in col_vals]
data_style +=[
        {
        'if': {'row_index': 'odd'},
        'backgroundColor': '#ffc5a3'
    },
        {
        'if': {'row_index': 'even'},
        'backgroundColor': '#f4f0ed'
    }
]

app.layout = html.Div(
    style ={'backgroundColor': '#f4f0ed'}, 
    children = [
        html.H1('FDIC Regulations Lookup', style = {'color':'#D71e28', 'fontFamily':'Arial', 'fontWeight': 'normal'}),
        dcc.Input(id='input1-box', type='text', value='', placeholder = 'Enter question here', style={'width': '50%', 'fontSize': '18px'}),
        html.Button('Submit', className='fancy-button', id='submit-button', n_clicks=0, style ={'margin-left': '10px'}),
        html.Div(id = 'table-container', style = {'display': 'none'}, children = [
        dash_table.DataTable(
                id = 'output-container',
                style_table = {'width': '100%', 'borderSpacing': '5px'},
                style_cell = {
                    'width': {
                            'index': '5%',
                            'Section': '18%',
                            'Subject': '12%',
                            'Similarity_Score': '5%',
                            'Text': '60%',
                        },
                    'fontSize': '16px',
                    'fontFamily': 'Arial',
                    'padding':'5px'
                    },
                columns = [
                    {'name': '#', 'id': 'index'},
                    {'name': 'Section', 'id': 'Section'},
                    {'name': 'Subject', 'id': 'Subject'},
                    {'name': 'Similarity', 'id': 'Similarity_Score'},
                    {'name': 'Text', 'id': 'Text'},
                ],
                data = [],
                style_data = {
                    'textAlign': 'left',
                    'lineHeight': '15px'
                },
                style_header = {
                    'text-align': 'left',
                    'fontSize': '20px',
                    'fontFamily':'Arial'
                },
                style_data_conditional=data_style,
        )
        ])
    ]
)


@app.callback(Output('table-container', 'style'),
              [Input('submit-button', 'n_clicks')])
def show_table(n_clicks):
    if n_clicks > 0:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(Output('output-container', 'data'),
              [Input('submit-button', 'n_clicks')],
              [State('input1-box', 'value')])
def update_output(n_clicks, value):
    if n_clicks > 0:
        return ask_question(value, vectors, df, roberta)
    else:
        return []

if __name__ == '__main__':
    app.run_server(debug=True)