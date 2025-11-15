import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
sys.path.append('../..')
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and prepare data from data pipeline
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = config.get('paths.data_dir')
        self.data_path = Path(data_path)
        
    def load_training_data(self, filename='complete_training_dataset.csv'):
        """
        Load training dataset from data pipeline
        """
        logger.info(f"Loading data from {filename}...")
        
        filepath = self.data_path / filename
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Training data not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        logger.info(f"✓ Loaded {len(df):,} records")
        logger.info(f"  Stocks: {df['symbol'].nunique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Features: {len(df.columns)}")
        
        return df
    
    def get_stock_data(self, df, symbol):
        """
        Get data for a specific stock
        """
        return df[df['symbol'] == symbol].copy()
    
    def split_by_date(self, df, train_end_date, val_end_date):
        """
        Split data by date (important for time series)
        """
        train_df = df[df['date'] <= train_end_date].copy()
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
        test_df = df[df['date'] > val_end_date].copy()
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_df):,} records ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"  Val:   {len(val_df):,} records ({val_df['date'].min()} to {val_df['date'].max()})")
        logger.info(f"  Test:  {len(test_df):,} records ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def calculate_future_returns(self, df, horizon_days=30):
        """
        Calculate future returns for labeling
        """
        logger.info(f"Calculating {horizon_days}-day future returns...")
        
        df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Calculate future returns for each stock
        df['future_return'] = df.groupby('symbol')['close'].shift(-horizon_days)
        df['future_return'] = ((df['future_return'] - df['close']) / df['close']) * 100
        
        # Remove rows without future returns (last N days of each stock)
        initial_len = len(df)
        df = df.dropna(subset=['future_return'])
        
        logger.info(f"✓ Calculated future returns")
        logger.info(f"  Removed {initial_len - len(df)} rows (no future data)")
        
        return df
    
    def create_labels(self, df):
        """
        Create classification labels based on future returns
        """
        logger.info("Creating classification labels...")
        
        strong_buy = config.get('labeling.strong_buy_threshold')
        buy = config.get('labeling.buy_threshold')
        hold_low = config.get('labeling.hold_threshold_low')
        sell = config.get('labeling.sell_threshold')
        
        conditions = [
            (df['future_return'] >= strong_buy),
            (df['future_return'] >= buy) & (df['future_return'] < strong_buy),
            (df['future_return'] >= hold_low) & (df['future_return'] < buy),
            (df['future_return'] >= sell) & (df['future_return'] < hold_low),
            (df['future_return'] < sell)
        ]
        
        choices = [4, 3, 2, 1, 0]  # Strong Buy, Buy, Hold, Sell, Strong Sell
        label_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        
        df['label'] = np.select(conditions, choices, default=2)
        df['label_name'] = df['label'].map(dict(enumerate(label_names)))
        
        # Log class distribution
        logger.info("Class distribution:")
        for i, name in enumerate(label_names):
            count = (df['label'] == i).sum()
            pct = (count / len(df)) * 100
            logger.info(f"  {name}: {count:,} ({pct:.1f}%)")
        
        return df
    
    def get_feature_columns(self, df, exclude_cols=None):
        """
        Get list of feature columns (exclude metadata and target)
        """
        if exclude_cols is None:
            exclude_cols = ['symbol', 'date', 'future_return', 'label', 'label_name', 
                          'open', 'high', 'low', 'close', 'volume']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        return feature_cols


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    
    # Load data
    df = loader.load_training_data()
    
    # Calculate future returns
    df = loader.calculate_future_returns(df, horizon_days=30)
    
    # Create labels
    df = loader.create_labels(df)
    
    # Get feature columns
    feature_cols = loader.get_feature_columns(df)
    
    print("\nSample data:")
    print(df[['symbol', 'date', 'close', 'future_return', 'label', 'label_name']].head(10))
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")