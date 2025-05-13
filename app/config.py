import os
from dotenv import load_dotenv
from datetime import datetime, time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API Keys with validation
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    logger.error("Missing Alpaca API credentials. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")
    raise ValueError("Missing Alpaca API credentials")

# Trading Settings
TRADING_HOURS = {
    'start': time(9, 30),  # 9:30 AM EST
    'end': time(16, 0)     # 4:00 PM EST
}

# Model Settings
HISTORICAL_DATA_YEARS = 5
TRAINING_INTERVAL_HOURS = 24  # Retrain model every 24 hours
PREDICTION_INTERVAL_MINUTES = 5  # Make predictions every 5 minutes
MIN_TRAINING_SAMPLES = 1000  # Minimum samples required for training
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for trade execution

# Risk Management (More conservative settings)
MAX_POSITION_SIZE = 0.05  # Maximum 5% of portfolio in single position (reduced from 10%)
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE = 0.04  # 4% take profit (reduced from 5%)
MAX_DAILY_TRADES = 5  # Maximum number of trades per day
MAX_OPEN_POSITIONS = 3  # Maximum number of open positions at once
MAX_DAILY_LOSS = 0.02  # Maximum 2% daily loss before stopping trading

# Dashboard Settings
DASHBOARD_PORT = 8050
API_PORT = 8000
UPDATE_INTERVAL_SECONDS = 5

# Data Storage
DATA_DIR = 'data'
MODEL_DIR = 'models'
HISTORICAL_DATA_DIR = os.path.join(DATA_DIR, 'historical')
REAL_TIME_DATA_DIR = os.path.join(DATA_DIR, 'realtime')
BACKUP_DIR = os.path.join(DATA_DIR, 'backup')

# Create necessary directories
for directory in [DATA_DIR, MODEL_DIR, HISTORICAL_DATA_DIR, REAL_TIME_DATA_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Model Parameters
MODEL_PARAMS = {
    'n_estimators': 200,  # Increased from 100
    'max_depth': 15,      # Increased from 10
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'random_state': 42
}

# Technical Indicators
TECHNICAL_INDICATORS = {
    'RSI': {'window': 14},
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'SMA': {'windows': [20, 50, 200]},
    'EMA': {'windows': [12, 26]},
    'Bollinger': {'window': 20, 'std': 2},
    'ATR': {'window': 14}
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'trading_system.log',
            'formatter': 'standard'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 