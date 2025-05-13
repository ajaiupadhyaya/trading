import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import ta
import pandas_ta as pta

class AdvancedTradingModel:
    def __init__(self):
        self.price_model = self._build_price_model()
        self.volatility_model = self._build_volatility_model()
        self.ensemble_model = self._build_ensemble_model()
        self.scaler = StandardScaler()
        
    def _build_price_model(self):
        # Price prediction model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(60, 12)),
            BatchNormalization(),
            Dropout(0.4),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.4),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def _build_volatility_model(self):
        # Volatility prediction model
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(60, 8)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        return model
        
    def _build_ensemble_model(self):
        # Ensemble model combining price and volatility predictions
        price_input = Input(shape=(1,))
        vol_input = Input(shape=(1,))
        
        combined = Concatenate()([price_input, vol_input])
        x = Dense(32, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dense(16, activation='relu')(x)
        output = Dense(3, activation='softmax')(x)  # 3 classes: hold, buy call, buy put
        
        model = Model(inputs=[price_input, vol_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
        
    def prepare_features(self, symbol, timeframe='1D', limit=100):
        # Get historical data
        end = datetime.now()
        start = end - timedelta(days=limit)
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start, end=end)
        
        # Create DataFrame with technical indicators
        df = pd.DataFrame()
        
        # Price and volume features
        df['close'] = hist['Close']
        df['volume'] = hist['Volume']
        df['high'] = hist['High']
        df['low'] = hist['Low']
        df['returns'] = df['close'].pct_change()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'])
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Create sequences for LSTM
        X_price = []
        X_vol = []
        y_price = []
        y_vol = []
        
        for i in range(60, len(df)):
            # Price prediction features
            price_features = df.iloc[i-60:i][
                ['close', 'volume', 'high', 'low', 'returns', 'sma_20', 
                 'ema_20', 'macd', 'rsi', 'stoch', 'obv', 'mfi']
            ].values
            X_price.append(price_features)
            
            # Volatility prediction features
            vol_features = df.iloc[i-60:i][
                ['returns', 'bb_high', 'bb_low', 'atr', 'rsi', 
                 'stoch', 'volume', 'mfi']
            ].values
            X_vol.append(vol_features)
            
            # Price target (next day's return)
            future_return = df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1
            y_price.append(future_return)
            
            # Volatility target (high volatility = 1, low volatility = 0)
            volatility = df.iloc[i]['atr'] / df.iloc[i]['close']
            y_vol.append(1 if volatility > df['atr'].mean() else 0)
            
        return (np.array(X_price), np.array(X_vol), 
                np.array(y_price), np.array(y_vol))
        
    def train(self, symbol):
        X_price, X_vol, y_price, y_vol = self.prepare_features(symbol)
        
        # Scale the features
        X_price_scaled = self.scaler.fit_transform(X_price.reshape(-1, X_price.shape[-1])).reshape(X_price.shape)
        X_vol_scaled = self.scaler.transform(X_vol.reshape(-1, X_vol.shape[-1])).reshape(X_vol.shape)
        
        # Train price model
        self.price_model.fit(
            X_price_scaled, y_price,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0
        )
        
        # Train volatility model
        self.volatility_model.fit(
            X_vol_scaled, y_vol,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0
        )
        
        # Prepare ensemble training data
        price_pred = self.price_model.predict(X_price_scaled)
        vol_pred = self.volatility_model.predict(X_vol_scaled)
        
        # Create ensemble targets
        y_ensemble = np.zeros((len(y_price), 3))
        for i in range(len(y_price)):
            if abs(y_price[i]) < 0.01:  # Small price movement
                y_ensemble[i, 0] = 1  # Hold
            elif y_price[i] > 0:
                y_ensemble[i, 1] = 1  # Buy call
            else:
                y_ensemble[i, 2] = 1  # Buy put
                
        # Train ensemble model
        self.ensemble_model.fit(
            [price_pred, vol_pred], y_ensemble,
            epochs=50, batch_size=32,
            validation_split=0.2, verbose=0
        )
        
    def predict(self, symbol):
        X_price, X_vol, _, _ = self.prepare_features(symbol, limit=61)
        
        # Scale the features
        X_price_scaled = self.scaler.transform(X_price.reshape(-1, X_price.shape[-1])).reshape(X_price.shape)
        X_vol_scaled = self.scaler.transform(X_vol.reshape(-1, X_vol.shape[-1])).reshape(X_vol.shape)
        
        # Get predictions
        price_pred = self.price_model.predict(X_price_scaled[-1:])
        vol_pred = self.volatility_model.predict(X_vol_scaled[-1:])
        
        # Get ensemble prediction
        ensemble_pred = self.ensemble_model.predict([price_pred, vol_pred])
        
        return {
            'price_prediction': float(price_pred[0][0]),
            'volatility_prediction': float(vol_pred[0][0]),
            'action_probabilities': ensemble_pred[0].tolist(),
            'recommended_action': np.argmax(ensemble_pred[0])
        }
        
    def should_trade(self, symbol):
        predictions = self.predict(symbol)
        action = predictions['recommended_action']
        confidence = predictions['action_probabilities'][action]
        
        # Only trade if confidence is high enough and volatility is favorable
        if confidence > 0.7 and predictions['volatility_prediction'] < 0.8:
            if action == 1:
                return True, 'call'
            elif action == 2:
                return True, 'put'
        return False, None 