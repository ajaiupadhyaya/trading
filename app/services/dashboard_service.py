import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class DashboardService:
    def __init__(self):
        self.portfolio_history = []
        self.trade_history = []
        self.model_metrics = {}
        
    def update_portfolio_value(self, value: float, timestamp: datetime = None):
        """Update portfolio value history"""
        if timestamp is None:
            timestamp = datetime.now()
        self.portfolio_history.append({
            'timestamp': timestamp,
            'value': value
        })
        
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a new trade to history"""
        trade_data['timestamp'] = datetime.now()
        self.trade_history.append(trade_data)
        
    def update_model_metrics(self, metrics: Dict[str, float]):
        """Update model performance metrics"""
        self.model_metrics = metrics
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for dashboard updates"""
        return {
            'portfolio_history': self._get_portfolio_data(),
            'trade_history': self._get_recent_trades(),
            'model_metrics': self.model_metrics,
            'summary_metrics': self._calculate_summary_metrics()
        }
    
    def _get_portfolio_data(self) -> List[Dict[str, Any]]:
        """Get formatted portfolio history"""
        return [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'value': entry['value']
            }
            for entry in self.portfolio_history
        ]
    
    def _get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent trades"""
        return sorted(
            self.trade_history,
            key=lambda x: x['timestamp'],
            reverse=True
        )[:limit]
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics for dashboard"""
        if not self.trade_history:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'active_positions': 0
            }
            
        # Calculate total P&L
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        win_rate = (winning_trades / len(self.trade_history)) * 100
        
        # Count active positions
        active_positions = sum(1 for trade in self.trade_history 
                             if trade['status'] == 'open')
        
        return {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'active_positions': active_positions
        } 