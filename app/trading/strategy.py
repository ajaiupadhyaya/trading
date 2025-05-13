import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

class MLTradingStrategy:
    def __init__(self, api):
        self.api = api
        self.model = self._build_model()
        self.scaler = StandardScaler()
        
    def _build_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def prepare_data(self, symbol, timeframe='1D', limit=100):
        # Get historical data
        end = datetime.now()
        start = end - timedelta(days=limit)
        bars = self.api.get_bars(symbol, timeframe, start.isoformat(), end.isoformat()).df
        
        # Prepare features
        df = pd.DataFrame()
        df['close'] = bars['close']
        df['volume'] = bars['volume']
        df['high'] = bars['high']
        df['low'] = bars['low']
        df['returns'] = df['close'].pct_change()
        
        # Create sequences for LSTM
        X = []
        y = []
        for i in range(60, len(df)):
            X.append(df.iloc[i-60:i][['close', 'volume', 'high', 'low', 'returns']].values)
            y.append(1 if df.iloc[i]['close'] > df.iloc[i-1]['close'] else 0)
            
        return np.array(X), np.array(y)
    
    def train(self, symbol):
        X, y = self.prepare_data(symbol)
        X = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    def predict(self, symbol):
        X, _ = self.prepare_data(symbol, limit=61)
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        prediction = self.model.predict(X[-1:])
        return prediction[0][0]
    
    def should_trade(self, symbol):
        prediction = self.predict(symbol)
        return prediction > 0.7  # Buy if prediction > 70% confidence
    
    def execute_trade(self, symbol):
        if self.should_trade(symbol):
            try:
                # Get current position
                position = self.api.get_position(symbol) if self.api.get_position(symbol) else None
                
                if position is None:
                    # Calculate position size (1% of portfolio)
                    account = self.api.get_account()
                    portfolio_value = float(account.portfolio_value)
                    price = float(self.api.get_latest_trade(symbol).price)
                    qty = int((portfolio_value * 0.01) / price)
                    
                    if qty > 0:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                        return f"Bought {qty} shares of {symbol}"
            except Exception as e:
                return f"Error executing trade: {str(e)}"
        return "No trade executed" 