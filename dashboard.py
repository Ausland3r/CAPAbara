import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# Загрузка данных
data_with_predictions = pd.read_csv('repository_data_with_predictions.csv')
recommendations_df = pd.read_csv('recommendations.csv')
feature_importances_df = pd.read_csv('feature_importances.csv')
author_analysis_df = pd.read_csv('author_analysis.csv')

repos = data_with_predictions['Repo'].unique()

# Создание Dash приложения
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("GitHub Commit Analysis Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Dropdown(
            id='repo-dropdown',
            options=[{'label': repo, 'value': repo} for repo in repos],
            value=repos[0],
            clearable=False
        ), width=4)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='additions-graph'), width=6),
        dbc.Col(dcc.Graph(id='deletions-graph'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='total-changes-graph'), width=6),
        dbc.Col(dcc.Graph(id='file-changes-graph'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='time-since-commit-graph'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='feature-importances-graph'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='clustering-additions-graph'), width=6),
        dbc.Col(dcc.Graph(id='clustering-deletions-graph'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='clustering-total-changes-graph'), width=6),
        dbc.Col(dcc.Graph(id='clustering-file-changes-graph'), width=6),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='clustering-time-since-commit-graph'), width=12),
    ]),
    dbc.Row([
        dbc.Col(html.H2("CAPA Recommendations"), className="mt-4")
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='capa-recommendations'), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.H2("Author Analysis"), className="mt-4")
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='author-activity-graph'), width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='author-capa-rate-graph'), width=12),
    ])
], fluid=True)

@app.callback(
    [Output('additions-graph', 'figure'),
     Output('deletions-graph', 'figure'),
     Output('total-changes-graph', 'figure'),
     Output('file-changes-graph', 'figure'),
     Output('time-since-commit-graph', 'figure'),
     Output('feature-importances-graph', 'figure'),
     Output('clustering-additions-graph', 'figure'),
     Output('clustering-deletions-graph', 'figure'),
     Output('clustering-total-changes-graph', 'figure'),
     Output('clustering-file-changes-graph', 'figure'),
     Output('clustering-time-since-commit-graph', 'figure'),
     Output('capa-recommendations', 'children'),
     Output('author-activity-graph', 'figure'),
     Output('author-capa-rate-graph', 'figure')],
    [Input('repo-dropdown', 'value')]
)
def update_dashboard(selected_repo):
    repo_data = data_with_predictions[data_with_predictions['Repo'] == selected_repo]

    fig_additions = px.histogram(repo_data, x='Additions', nbins=30, title='Additions Distribution', color_discrete_sequence=['#636EFA'])
    fig_deletions = px.histogram(repo_data, x='Deletions', nbins=30, title='Deletions Distribution', color_discrete_sequence=['#EF553B'])
    fig_total_changes = px.histogram(repo_data, x='Total Changes', nbins=30, title='Total Changes Distribution', color_discrete_sequence=['#00CC96'])
    fig_file_changes = px.histogram(repo_data, x='File Changes', nbins=30, title='File Changes Distribution', color_discrete_sequence=['#AB63FA'])
    fig_time_since_commit = px.histogram(repo_data, x='Time Since Last Commit', nbins=30, title='Time Since Last Commit Distribution', color_discrete_sequence=['#FFA15A'])

    fig_feature_importances = px.bar(feature_importances_df, x='Importance', y='Feature', orientation='h', title='RandomForest Feature Importances', color='Feature', color_discrete_sequence=px.colors.qualitative.Plotly)

    fig_clustering_additions = px.scatter(repo_data, x=repo_data.index, y='Additions', color='Cluster', title='Additions Clustering', color_continuous_scale='Viridis')
    fig_clustering_additions.add_hline(y=repo_data['Additions'].mean(), line_dash="dash", line_color="red")

    fig_clustering_deletions = px.scatter(repo_data, x=repo_data.index, y='Deletions', color='Cluster', title='Deletions Clustering', color_continuous_scale='Viridis')
    fig_clustering_deletions.add_hline(y=repo_data['Deletions'].mean(), line_dash="dash", line_color="red")

    fig_clustering_total_changes = px.scatter(repo_data, x=repo_data.index, y='Total Changes', color='Cluster', title='Total Changes Clustering', color_continuous_scale='Viridis')
    fig_clustering_total_changes.add_hline(y=repo_data['Total Changes'].mean(), line_dash="dash", line_color="red")

    fig_clustering_file_changes = px.scatter(repo_data, x=repo_data.index, y='File Changes', color='Cluster', title='File Changes Clustering', color_continuous_scale='Viridis')
    fig_clustering_file_changes.add_hline(y=repo_data['File Changes'].mean(), line_dash="dash", line_color="red")

    fig_clustering_time_since_commit = px.scatter(repo_data, x=repo_data.index, y='Time Since Last Commit', color='Cluster', title='Time Since Last Commit Clustering', color_continuous_scale='Viridis')
    fig_clustering_time_since_commit.add_hline(y=repo_data['Time Since Last Commit'].mean(), line_dash="dash", line_color="red")

    capa_recs = recommendations_df[(recommendations_df['Repo'] == selected_repo) & (recommendations_df['Suggestion'] != 'No specific recommendations')]
    capa_recs_html = [
        html.P(
            [
                html.A(f"{row['Commit SHA'][:7]}...", href=f"https://github.com/{selected_repo}/commit/{row['Commit SHA']}", target="_blank"),
                f": {row['Suggestion']}"
            ]
        ) for index, row in capa_recs.iterrows()
    ]

    author_data = author_analysis_df[author_analysis_df['Repo'] == selected_repo]

    fig_author_activity = px.bar(author_data, x='Author', y='Commit Count', title='Author Activity (Commit Count)', color='Author', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig_author_capa_rate = px.bar(author_data, x='Author', y='CAPA Rate', title='Author CAPA Rate', color='Author', color_discrete_sequence=px.colors.qualitative.Plotly)

    return (fig_additions, fig_deletions, fig_total_changes, fig_file_changes,
            fig_time_since_commit, fig_feature_importances, fig_clustering_additions,
            fig_clustering_deletions, fig_clustering_total_changes, fig_clustering_file_changes,
            fig_clustering_time_since_commit, capa_recs_html, fig_author_activity, fig_author_capa_rate)

if __name__ == '__main__':
    app.run_server(debug=True)
