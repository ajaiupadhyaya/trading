import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alpaca_connection():
    """Test connection to Alpaca API"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API credentials
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not api_key or not api_secret:
            raise ValueError("Missing API credentials in .env file")
            
        # Initialize API
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        
        # Test API connection
        account = api.get_account()
        
        # Print account information
        logger.info("Successfully connected to Alpaca API!")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        
        # Test market data access
        clock = api.get_clock()
        logger.info(f"Market Status: {'Open' if clock.is_open else 'Closed'}")
        if clock.is_open:
            logger.info(f"Next Market Open: {clock.next_open}")
        else:
            logger.info(f"Next Market Open: {clock.next_open}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {str(e)}")
        return False

if __name__ == "__main__":
    test_alpaca_connection() 