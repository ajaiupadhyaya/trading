import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Initialize the Dash app with a modern theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Layout components
def create_metric_card(title, value, change):
    return dbc.Card(
        dbc.CardBody([
            html.H4(title, className="card-title"),
            html.H2(value, className="card-value"),
            html.P(change, className="card-change")
        ]),
        className="metric-card"
    )

# Main layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Trading Dashboard", className="text-center my-4"), width=12)
    ]),
    
    # Key Metrics Row
    dbc.Row([
        dbc.Col(create_metric_card("Total P&L", "$0.00", "+0.00%"), width=3),
        dbc.Col(create_metric_card("Win Rate", "0%", "0%"), width=3),
        dbc.Col(create_metric_card("Model Accuracy", "0%", "0%"), width=3),
        dbc.Col(create_metric_card("Active Positions", "0", "0"), width=3),
    ], className="mb-4"),
    
    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Portfolio Performance"),
                dbc.CardBody(dcc.Graph(id='portfolio-performance'))
            ])
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Metrics"),
                dbc.CardBody(dcc.Graph(id='model-metrics'))
            ])
        ], width=4),
    ], className="mb-4"),
    
    # Recent Trades Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent Trades"),
                dbc.CardBody(html.Div(id='recent-trades-table'))
            ])
        ], width=12)
    ]),
    
    # Hidden div for storing data
    dcc.Store(id='trading-data'),
    
    # Interval component for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # Update every 5 seconds
        n_intervals=0
    )
], fluid=True)

# Callback to update portfolio performance chart
@app.callback(
    Output('portfolio-performance', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('trading-data', 'data')
)
def update_portfolio_performance(n, data):
    # This will be connected to your actual trading data
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.date_range(start='2024-01-01', periods=100),
        y=np.random.normal(0, 1, 100).cumsum(),
        name='Portfolio Value',
        line=dict(color='#00ff00')
    ))
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    return fig

# Callback to update model metrics
@app.callback(
    Output('model-metrics', 'figure'),
    Input('interval-component', 'n_intervals'),
    Input('trading-data', 'data')
)
def update_model_metrics(n, data):
    # This will be connected to your actual model metrics
    metrics = {
        'Accuracy': 0.75,
        'Precision': 0.68,
        'Recall': 0.72,
        'F1 Score': 0.70
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            marker_color='#00ff00'
        )
    ])
    fig.update_layout(
        template='plotly_dark',
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(range=[0, 1])
    )
    return fig

# Callback to update recent trades table
@app.callback(
    Output('recent-trades-table', 'children'),
    Input('interval-component', 'n_intervals'),
    Input('trading-data', 'data')
)
def update_recent_trades(n, data):
    # This will be connected to your actual trade data
    trades = pd.DataFrame({
        'Symbol': ['AAPL', 'GOOGL', 'MSFT'],
        'Type': ['BUY', 'SELL', 'BUY'],
        'Price': [150.25, 2800.75, 300.50],
        'Quantity': [10, 5, 15],
        'P&L': [25.50, -15.25, 45.75]
    })
    
    return dbc.Table.from_dataframe(
        trades,
        striped=True,
        bordered=True,
        hover=True,
        dark=True
    )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 