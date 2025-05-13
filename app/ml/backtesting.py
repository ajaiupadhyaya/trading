import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from .models import AdvancedTradingModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Backtester:
    def __init__(self, model, initial_capital=100000):
        self.model = model
        self.initial_capital = initial_capital
        self.results = []
        
    def run_backtest(self, symbol, start_date, end_date):
        # Get historical data
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = None
        trades = []
        
        # Run simulation
        for i in range(60, len(hist)):
            current_date = hist.index[i]
            
            # Get model prediction
            predictions = self.model.predict(symbol)
            action = predictions['recommended_action']
            confidence = predictions['action_probabilities'][action]
            
            # Calculate returns
            if position is not None:
                # Simulate options returns
                if position['type'] == 'call':
                    price_change = (hist['Close'][i] - position['entry_price']) / position['entry_price']
                    returns = price_change * 100  # Options have higher leverage
                else:
                    price_change = (position['entry_price'] - hist['Close'][i]) / position['entry_price']
                    returns = price_change * 100
                
                # Update capital
                capital *= (1 + returns/100)
                
                # Record trade
                trades.append({
                    'date': current_date,
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': hist['Close'][i],
                    'returns': returns,
                    'capital': capital
                })
                
                position = None
            
            # Check for new trade
            if confidence > 0.7 and predictions['volatility_prediction'] < 0.8:
                if action == 1:  # Buy call
                    position = {
                        'type': 'call',
                        'entry_price': hist['Close'][i]
                    }
                elif action == 2:  # Buy put
                    position = {
                        'type': 'put',
                        'entry_price': hist['Close'][i]
                    }
        
        # Calculate performance metrics
        self.results = self._calculate_metrics(trades, capital)
        return self.results
    
    def _calculate_metrics(self, trades, final_capital):
        if not trades:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'trades': []
            }
        
        # Convert trades to DataFrame
        df_trades = pd.DataFrame(trades)
        
        # Calculate metrics
        total_return = (final_capital - self.initial_capital) / self.initial_capital * 100
        returns = df_trades['returns'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        # Calculate drawdown
        cumulative_returns = (1 + returns/100).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / running_max
        max_drawdown = np.max(drawdown) * 100
        
        # Calculate win rate
        win_rate = len(returns[returns > 0]) / len(returns) * 100
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': trades
        }
    
    def plot_results(self):
        if not self.results['trades']:
            return None
            
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add capital line
        capital_data = pd.DataFrame(self.results['trades'])
        fig.add_trace(
            go.Scatter(
                x=capital_data['date'],
                y=capital_data['capital'],
                name="Portfolio Value"
            ),
            secondary_y=False
        )
        
        # Add trade markers
        for trade in self.results['trades']:
            color = 'green' if trade['returns'] > 0 else 'red'
            fig.add_trace(
                go.Scatter(
                    x=[trade['date']],
                    y=[trade['capital']],
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=10,
                        symbol='triangle-up' if trade['type'] == 'call' else 'triangle-down'
                    ),
                    name=f"{trade['type'].upper()} {trade['returns']:.1f}%"
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            showlegend=True
        )
        
        return fig 