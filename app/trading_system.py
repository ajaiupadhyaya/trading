import schedule
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Any
import json
import os

from .models.ml_model import MLTradingModel
from .services.data_service import DataService
from .services.dashboard_service import DashboardService
from .config import (
    TRAINING_INTERVAL_HOURS,
    PREDICTION_INTERVAL_MINUTES,
    MAX_POSITION_SIZE,
    STOP_LOSS_PERCENTAGE,
    TAKE_PROFIT_PERCENTAGE,
    MODEL_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self):
        self.data_service = DataService()
        self.dashboard_service = DashboardService()
        self.models: Dict[str, MLTradingModel] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.portfolio_value = 0.0
        
    def initialize(self):
        """Initialize the trading system"""
        logger.info("Initializing trading system...")
        
        # Initialize models for each symbol
        for symbol in self.data_service.get_trading_symbols():
            self.models[symbol] = MLTradingModel()
            
        # Load historical data and train models
        self.train_all_models()
        
        # Start scheduled tasks
        self._setup_schedules()
        
        logger.info("Trading system initialized successfully")
        
    def train_all_models(self):
        """Train all models with historical data"""
        logger.info("Training all models...")
        
        for symbol, model in self.models.items():
            try:
                # Get historical data
                data = self.data_service.get_historical_data(symbol)
                
                # Train model
                model.train(data)
                
                # Save model metrics
                metrics = model.get_model_metrics()
                self.dashboard_service.update_model_metrics(metrics)
                
                logger.info(f"Model trained for {symbol}")
                
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
                
    def make_predictions(self):
        """Make predictions for all symbols"""
        if not self.data_service.get_market_status():
            logger.info("Market is closed, skipping predictions")
            return
            
        logger.info("Making predictions...")
        
        for symbol, model in self.models.items():
            try:
                # Get latest data
                data = self.data_service.get_realtime_data(symbol)
                
                # Make prediction
                prediction = model.predict(data)
                
                # Execute trade if confidence is high enough
                if prediction['confidence'] > 0.7:  # 70% confidence threshold
                    self._execute_trade(symbol, prediction)
                    
            except Exception as e:
                logger.error(f"Error making prediction for {symbol}: {str(e)}")
                
    def _execute_trade(self, symbol: str, prediction: Dict[str, Any]):
        """Execute a trade based on prediction"""
        current_price = self.data_service.get_latest_price(symbol)
        
        # Calculate position size
        position_size = self.portfolio_value * MAX_POSITION_SIZE / current_price
        
        # Create trade data
        trade_data = {
            'symbol': symbol,
            'type': prediction['action'],
            'price': current_price,
            'quantity': int(position_size),
            'pnl': 0.0,
            'status': 'open',
            'stop_loss': current_price * (1 - STOP_LOSS_PERCENTAGE),
            'take_profit': current_price * (1 + TAKE_PROFIT_PERCENTAGE)
        }
        
        # Update positions
        self.positions[symbol] = trade_data
        
        # Update dashboard
        self.dashboard_service.add_trade(trade_data)
        
        logger.info(f"Executed {prediction['action']} trade for {symbol}")
        
    def check_positions(self):
        """Check and update open positions"""
        for symbol, position in self.positions.items():
            if position['status'] == 'open':
                current_price = self.data_service.get_latest_price(symbol)
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    self._close_position(symbol, current_price, 'stop_loss')
                    
                # Check take profit
                elif current_price >= position['take_profit']:
                    self._close_position(symbol, current_price, 'take_profit')
                    
    def _close_position(self, symbol: str, current_price: float, reason: str):
        """Close an open position"""
        position = self.positions[symbol]
        
        # Calculate P&L
        pnl = (current_price - position['price']) * position['quantity']
        if position['type'] == 'SELL':
            pnl = -pnl
            
        # Update position
        position['status'] = 'closed'
        position['pnl'] = pnl
        position['close_price'] = current_price
        position['close_reason'] = reason
        
        # Update dashboard
        self.dashboard_service.add_trade(position)
        
        logger.info(f"Closed position for {symbol} due to {reason}")
        
    def _setup_schedules(self):
        """Setup scheduled tasks"""
        # Train models periodically
        schedule.every(TRAINING_INTERVAL_HOURS).hours.do(self.train_all_models)
        
        # Make predictions periodically
        schedule.every(PREDICTION_INTERVAL_MINUTES).minutes.do(self.make_predictions)
        
        # Check positions every minute
        schedule.every(1).minutes.do(self.check_positions)
        
    def run(self):
        """Run the trading system"""
        logger.info("Starting trading system...")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in trading system: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying
                
if __name__ == "__main__":
    trading_system = TradingSystem()
    trading_system.initialize()
    trading_system.run() 