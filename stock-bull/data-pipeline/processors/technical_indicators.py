import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for stock price data
    """
    
    @staticmethod
    def calculate_sma(df, periods=[20, 50, 200]):
        """Simple Moving Average"""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_ema(df, periods=[12, 26]):
        """Exponential Moving Average"""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def calculate_rsi(df, period=14):
        """Relative Strength Index"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def calculate_macd(df, fast=12, slow=26, signal=9):
        """Moving Average Convergence Divergence"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def calculate_bollinger_bands(df, period=20, std=2):
        """Bollinger Bands"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        df['bb_std'] = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (std * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (std * df['bb_std'])
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        return df
    
    @staticmethod
    def calculate_atr(df, period=14):
        """Average True Range (Volatility)"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        return df
    
    @staticmethod
    def calculate_momentum(df, periods=[5, 10, 20]):
        """Price Momentum"""
        for period in periods:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'momentum_pct_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        return df
    
    @staticmethod
    def calculate_volume_indicators(df):
        """Volume-based indicators"""
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    @staticmethod
    def calculate_price_levels(df):
        """Support/Resistance levels"""
        # 52-week high/low
        df['52w_high'] = df['high'].rolling(window=252).max()
        df['52w_low'] = df['low'].rolling(window=252).min()
        
        # Distance from 52-week high/low
        df['dist_from_52w_high'] = ((df['close'] - df['52w_high']) / df['52w_high']) * 100
        df['dist_from_52w_low'] = ((df['close'] - df['52w_low']) / df['52w_low']) * 100
        
        return df
    
    @staticmethod
    def calculate_all_indicators(df):
        """
        Calculate all technical indicators
        Input: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        """
        logger.info("Calculating technical indicators...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Moving Averages
        df = TechnicalIndicators.calculate_sma(df)
        df = TechnicalIndicators.calculate_ema(df)
        
        # Momentum Indicators
        df = TechnicalIndicators.calculate_rsi(df)
        df = TechnicalIndicators.calculate_macd(df)
        df = TechnicalIndicators.calculate_momentum(df)
        
        # Volatility Indicators
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        df = TechnicalIndicators.calculate_atr(df)
        
        # Volume Indicators
        df = TechnicalIndicators.calculate_volume_indicators(df)
        
        # Price Levels
        df = TechnicalIndicators.calculate_price_levels(df)
        
        # Additional derived features
        df['high_low_pct'] = ((df['high'] - df['low']) / df['low']) * 100
        df['close_open_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        
        # Trend indicators
        df['sma_20_50_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['price_above_sma_200'] = np.where(df['close'] > df['sma_200'], 1, 0)
        
        logger.info(f"✓ Calculated {len(df.columns) - 6} technical indicators")
        
        return df


# Test the indicators
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100)
    sample_data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000000, 2000000, 100)
    })
    
    # Calculate indicators
    result = TechnicalIndicators.calculate_all_indicators(sample_data)
    
    print("\nSample Technical Indicators:")
    print(result[['date', 'close', 'sma_20', 'sma_50', 'rsi', 'macd']].tail(10))
    print(f"\nTotal features: {len(result.columns)}")
    print(f"Feature names: {result.columns.tolist()}")