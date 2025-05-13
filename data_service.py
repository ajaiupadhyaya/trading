import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import time

SYMBOLS = ['AAPL', 'MSFT', 'SPY']
DATA_DIR = 'data'
YEARS = 5

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def fetch_from_alpaca(symbol, years=YEARS, test_one_year=False):
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    if not api_key or not api_secret:
        print('Missing Alpaca API credentials.')
        return None
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    end = datetime.now()
    start = end - timedelta(days=365*(1 if test_one_year else years))
    all_bars = []
    current_start = start
    print(f"Fetching Alpaca data for {symbol} from {start.date()} to {end.date()}...")
    while current_start < end:
        current_end = min(current_start + timedelta(days=1000), end)
        try:
            barset = api.get_bars(symbol, '1D', start=current_start.isoformat(), end=current_end.isoformat(), adjustment='raw', limit=1000)
            bars = [bar._raw for bar in barset]
            if not bars:
                break
            all_bars.extend(bars)
            last_time = pd.to_datetime(bars[-1]['t'])
            current_start = last_time + timedelta(days=1)
            time.sleep(0.25)  # avoid rate limits
        except Exception as e:
            print(f"Alpaca error for {symbol}: {e}")
            break
    if all_bars:
        df = pd.DataFrame(all_bars)
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        df.to_csv(f'{DATA_DIR}/{symbol}.csv')
        print(f"Saved {symbol} data from Alpaca to {DATA_DIR}/{symbol}.csv ({len(df)} rows)")
        return df
    else:
        print(f"No data for {symbol} from Alpaca.")
        return None

def fetch_and_save(symbol, years=YEARS, test_one_year=False):
    end = datetime.now()
    start = end - timedelta(days=365*(1 if test_one_year else years))
    try:
        print(f"Trying yfinance for {symbol}...")
        df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if not df.empty:
            df.to_csv(f'{DATA_DIR}/{symbol}.csv')
            print(f"Saved {symbol} data from yfinance to {DATA_DIR}/{symbol}.csv ({len(df)} rows)")
            return df
        else:
            print(f"No data for {symbol} from yfinance, trying Alpaca...")
            return fetch_from_alpaca(symbol, years, test_one_year)
    except Exception as e:
        print(f"yfinance error for {symbol}: {e}, trying Alpaca...")
        return fetch_from_alpaca(symbol, years, test_one_year)

def update_all_symbols(years=YEARS, test_one_year=False):
    ensure_data_dir()
    for symbol in SYMBOLS:
        fetch_and_save(symbol, years, test_one_year)

def test_all_systems():
    print("\n--- TEST: Fetching 1 year of data for all symbols ---")
    update_all_symbols(years=1, test_one_year=True)
    print("\n--- TEST: Fetching full 5 years of data for all symbols ---")
    update_all_symbols(years=YEARS, test_one_year=False)

def test_yfinance_single_day(symbol='AAPL'):
    print(f"\nTesting yfinance for {symbol} (single day)...")
    today = datetime.now().date()
    df = yf.download(symbol, start=str(today), end=str(today + timedelta(days=1)))
    print(df.head() if not df.empty else f"No data for {symbol} from yfinance on {today}")

def test_alpaca_single_day(symbol='AAPL'):
    print(f"\nTesting Alpaca for {symbol} (single day)...")
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    if not api_key or not api_secret:
        print('Missing Alpaca API credentials.')
        return
    api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
    today = datetime.now().date()
    try:
        bars = api.get_bars(symbol, '1Day', start=str(today), end=str(today + timedelta(days=1)), adjustment='raw', limit=10)
        bars = [bar._raw for bar in bars]
        if bars:
            df = pd.DataFrame(bars)
            print(df.head())
        else:
            print(f"No data for {symbol} from Alpaca on {today}")
    except Exception as e:
        print(f"Alpaca error for {symbol} (single day): {e}")

if __name__ == "__main__":
    test_yfinance_single_day('AAPL')
    test_alpaca_single_day('AAPL')
    test_all_systems() 