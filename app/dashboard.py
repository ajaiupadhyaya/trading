import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime, timedelta
from .ml.models import AdvancedTradingModel
from .ml.backtesting import Backtester
from .ml.sentiment import MarketSentimentAnalyzer

# Initialize components
model = AdvancedTradingModel()
sentiment_analyzer = MarketSentimentAnalyzer()

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("AI Trading Bot Dashboard", className="text-center mb-4"),
            dbc.Card([
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.Input(id="symbol-input", placeholder="Enter stock symbol...", type="text"),
                        dbc.Button("Analyze", id="analyze-button", color="primary")
                    ])
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Trading Signals"),
                dbc.CardBody([
                    dcc.Graph(id="signals-graph")
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Market Sentiment"),
                dbc.CardBody([
                    dcc.Graph(id="sentiment-graph")
                ])
            ])
        ], width=6)
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Backtest Results"),
                dbc.CardBody([
                    dcc.Graph(id="backtest-graph")
                ])
            ])
        ])
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Metrics"),
                dbc.CardBody(id="metrics-table")
            ])
        ])
    ], className="mt-4")
])

# Callbacks
@app.callback(
    [Output("signals-graph", "figure"),
     Output("sentiment-graph", "figure"),
     Output("backtest-graph", "figure"),
     Output("metrics-table", "children")],
    [Input("analyze-button", "n_clicks")],
    [State("symbol-input", "value")]
)
def update_dashboard(n_clicks, symbol):
    if not symbol:
        return {}, {}, {}, "Please enter a symbol"
        
    # Get predictions
    predictions = model.predict(symbol)
    
    # Get sentiment
    sentiment = sentiment_analyzer.get_market_sentiment(symbol)
    
    # Run backtest
    backtester = Backtester(model)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    backtest_results = backtester.run_backtest(symbol, start_date, end_date)
    
    # Create signals graph
    signals_fig = go.Figure()
    signals_fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=predictions['action_probabilities'][predictions['recommended_action']] * 100,
        title={'text': "Trading Signal Confidence"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 30], 'color': "red"},
                   {'range': [30, 70], 'color': "yellow"},
                   {'range': [70, 100], 'color': "green"}
               ]}
    ))
    
    # Create sentiment graph
    sentiment_fig = go.Figure()
    if sentiment:
        sentiment_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sentiment['overall_sentiment'] * 100,
            title={'text': "Market Sentiment"},
            gauge={'axis': {'range': [-100, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [-100, -20], 'color': "red"},
                       {'range': [-20, 20], 'color': "yellow"},
                       {'range': [20, 100], 'color': "green"}
                   ]}
        ))
    
    # Create backtest graph
    backtest_fig = backtester.plot_results()
    
    # Create metrics table
    metrics_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Metric"),
            html.Th("Value")
        ])),
        html.Tbody([
            html.Tr([html.Td("Total Return"), html.Td(f"{backtest_results['total_return']:.2f}%")]),
            html.Tr([html.Td("Sharpe Ratio"), html.Td(f"{backtest_results['sharpe_ratio']:.2f}")]),
            html.Tr([html.Td("Max Drawdown"), html.Td(f"{backtest_results['max_drawdown']:.2f}%")]),
            html.Tr([html.Td("Win Rate"), html.Td(f"{backtest_results['win_rate']:.2f}%")])
        ])
    ], bordered=True, dark=True, hover=True, responsive=True)
    
    return signals_fig, sentiment_fig, backtest_fig, metrics_table

if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 