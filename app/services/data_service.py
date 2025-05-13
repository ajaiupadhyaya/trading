import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from ..config import HISTORICAL_DATA_DIR, REAL_TIME_DATA_DIR, HISTORICAL_DATA_YEARS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataService:
    def __init__(self):
        self.cache = {}
        
    def get_historical_data(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """Get historical data for a symbol"""
        cache_file = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}_historical.csv")
        
        if not force_refresh and os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded historical data for {symbol} from cache")
            return data
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * HISTORICAL_DATA_YEARS)
        
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval='1d'
            )
            
            # Save to cache
            data.to_csv(cache_file)
            logger.info(f"Downloaded and cached historical data for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading historical data for {symbol}: {str(e)}")
            raise
            
    def get_realtime_data(self, symbol: str) -> pd.DataFrame:
        """Get real-time data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            return data
        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            raise
            
    def update_realtime_data(self, symbol: str) -> None:
        """Update real-time data cache"""
        try:
            data = self.get_realtime_data(symbol)
            cache_file = os.path.join(REAL_TIME_DATA_DIR, f"{symbol}_realtime.csv")
            data.to_csv(cache_file)
            self.cache[symbol] = data
        except Exception as e:
            logger.error(f"Error updating real-time data for {symbol}: {str(e)}")
            
    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for a symbol"""
        if symbol not in self.cache:
            self.update_realtime_data(symbol)
        return self.cache[symbol]['Close'].iloc[-1]
        
    def get_market_status(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        return TRADING_HOURS['start'] <= now <= TRADING_HOURS['end']
        
    def get_trading_symbols(self) -> list:
        """Get list of symbols to trade"""
        # You can customize this list based on your trading strategy
        return [
            'SPY',  # S&P 500 ETF
            'QQQ',  # NASDAQ 100 ETF
            'AAPL', # Apple
            'MSFT', # Microsoft
            'GOOGL',# Google
            'AMZN', # Amazon
            'META', # Meta (Facebook)
            'TSLA', # Tesla
            'NVDA', # NVIDIA
            'AMD'   # Advanced Micro Devices
        ] 