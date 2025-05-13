import os
from dotenv import load_dotenv
from datetime import datetime, time

# Load environment variables
load_dotenv()

# API Keys
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')  # Use paper trading by default

# Trading Settings
TRADING_HOURS = {
    'start': time(9, 30),  # 9:30 AM EST
    'end': time(16, 0)     # 4:00 PM EST
}

# Model Settings
HISTORICAL_DATA_YEARS = 5
TRAINING_INTERVAL_HOURS = 24  # Retrain model every 24 hours
PREDICTION_INTERVAL_MINUTES = 5  # Make predictions every 5 minutes

# Risk Management
MAX_POSITION_SIZE = 0.1  # Maximum 10% of portfolio in single position
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.05  # 5% take profit

# Dashboard Settings
DASHBOARD_PORT = 8050
API_PORT = 8000
UPDATE_INTERVAL_SECONDS = 5

# Data Storage
DATA_DIR = 'data'
MODEL_DIR = 'models'
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, 'historical')
REAL_TIME_DATA_DIR = os.path.join(DATA_DIR, 'realtime')

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR, HISTORICAL_DATA_DIR, REAL_TIME_DATA_DIR]:
    os.makedirs(directory, exist_ok=True) 