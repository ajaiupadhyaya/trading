import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import yfinance as yf

class OptionsTradingStrategy:
    def __init__(self, api):
        self.api = api
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
    def _build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(60, 8)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(3, activation='softmax')  # 3 outputs: buy call, buy put, hold
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def calculate_implied_volatility(self, option_price, stock_price, strike_price, time_to_expiry, option_type):
        # Simplified Black-Scholes implied volatility calculation
        try:
            # This is a simplified version - in production, use a proper options pricing library
            if option_type == 'call':
                return (option_price / stock_price) * np.sqrt(2 * np.pi / time_to_expiry)
            else:
                return (option_price / strike_price) * np.sqrt(2 * np.pi / time_to_expiry)
        except:
            return None

    def prepare_data(self, symbol, timeframe='1D', limit=100):
        # Get historical data
        end = datetime.now()
        start = end - timedelta(days=limit)
        
        # Get stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start, end=end)
        
        # Get options data
        options = stock.options
        if not options:
            return None, None
            
        # Prepare features
        df = pd.DataFrame()
        df['close'] = hist['Close']
        df['volume'] = hist['Volume']
        df['high'] = hist['High']
        df['low'] = hist['Low']
        df['returns'] = df['close'].pct_change()
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Create sequences for LSTM
        X = []
        y = []
        
        for i in range(60, len(df)):
            features = df.iloc[i-60:i][
                ['close', 'volume', 'high', 'low', 'returns', 
                 'sma_20', 'rsi', 'volatility']
            ].values
            
            # Determine the target (0: hold, 1: buy call, 2: buy put)
            future_return = df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1
            if abs(future_return) < 0.01:  # Less than 1% movement
                target = 0  # Hold
            elif future_return > 0:
                target = 1  # Buy call
            else:
                target = 2  # Buy put
                
            X.append(features)
            y.append(target)
            
        return np.array(X), np.array(y)
    
    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def train(self, symbol):
        X, y = self.prepare_data(symbol)
        if X is None or y is None:
            return False
            
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((len(y), 3))
        y_one_hot[np.arange(len(y)), y] = 1
        
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.model.fit(X, y_one_hot, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        return True
    
    def predict(self, symbol):
        X, _ = self.prepare_data(symbol, limit=61)
        if X is None:
            return None
            
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        prediction = self.model.predict(X[-1:])
        return prediction[0]
    
    def should_trade(self, symbol):
        prediction = self.predict(symbol)
        if prediction is None:
            return False, None
            
        action = np.argmax(prediction)
        confidence = prediction[action]
        
        # Only trade if confidence is high enough
        if confidence > 0.7:
            if action == 1:
                return True, 'call'
            elif action == 2:
                return True, 'put'
        return False, None
    
    def execute_trade(self, symbol):
        should_trade, option_type = self.should_trade(symbol)
        if not should_trade:
            return "No trade executed"
            
        try:
            # Get current stock price
            stock = yf.Ticker(symbol)
            current_price = stock.info['regularMarketPrice']
            
            # Get available options
            options = stock.options
            if not options:
                return "No options available"
                
            # Find the nearest expiration date
            nearest_expiry = min(options, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
            
            # Get options chain
            opt = stock.option_chain(nearest_expiry)
            
            if option_type == 'call':
                options_data = opt.calls
            else:
                options_data = opt.puts
                
            # Find the closest strike price to current price
            strike_price = min(options_data['strike'], key=lambda x: abs(x - current_price))
            option = options_data[options_data['strike'] == strike_price].iloc[0]
            
            # Calculate position size (0.5% of portfolio)
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            option_price = float(option['lastPrice'])
            qty = int((portfolio_value * 0.005) / option_price)
            
            if qty > 0:
                self.api.submit_order(
                    symbol=option['symbol'],
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                return f"Bought {qty} {option_type} options of {symbol}"
                
        except Exception as e:
            return f"Error executing trade: {str(e)}"
            
        return "No trade executed" 