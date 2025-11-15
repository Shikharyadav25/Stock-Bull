import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import logging
import joblib
from pathlib import Path
import sys
sys.path.append('../..')
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocess data for ML models
    """
    
    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.feature_cols = None
        
    def handle_missing_values(self, df, method='forward_fill'):
        """
        Handle missing values in dataset
        """
        logger.info(f"Handling missing values using {method}...")
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_features = missing_counts[missing_counts > 0]
        
        if len(missing_features) > 0:
            logger.info(f"Found missing values in {len(missing_features)} features")
            
            if method == 'forward_fill':
                # Forward fill within each stock
                df = df.sort_values(['symbol', 'date'])
                df = df.groupby('symbol').fillna(method='ffill')
                df = df.fillna(method='bfill')  # Backward fill remaining
                
            elif method == 'interpolate':
                df = df.groupby('symbol').apply(lambda group: group.interpolate(method='linear'))
                
            elif method == 'drop':
                df = df.dropna()
                
            # Fill any remaining NaN with 0
            df = df.fillna(0)
            
        logger.info(f"✓ Missing values handled")
        return df
    
    def remove_outliers(self, df, feature_cols, threshold=3.0):
        """
        Remove outliers using IQR method
        """
        logger.info(f"Removing outliers (threshold: {threshold})...")
        
        initial_len = len(df)
        
        for col in feature_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        removed = initial_len - len(df)
        logger.info(f"✓ Removed {removed:,} outlier rows ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def scale_features(self, X_train, X_val=None, X_test=None, method='robust'):
        """
        Scale features using specified method
        """
        logger.info(f"Scaling features using {method} scaler...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {method}. Using RobustScaler.")
            self.scaler = RobustScaler()
        
        # Fit on training data only
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = {'train': X_train_scaled}
        
        if X_val is not None:
            results['val'] = self.scaler.transform(X_val)
        
        if X_test is not None:
            results['test'] = self.scaler.transform(X_test)
        
        logger.info("✓ Features scaled")
        
        return results
    
    def prepare_features_labels(self, df, feature_cols, target_col='label'):
        """
        Separate features and labels
        """
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Store metadata
        metadata = df[['symbol', 'date', 'close']].copy()
        
        return X, y, metadata
    
    def save_preprocessor(self, path='./models/preprocessor.pkl'):
        """
        Save preprocessor objects
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'scaler': self.scaler,
            'feature_cols': self.feature_cols
        }, path)
        
        logger.info(f"✓ Preprocessor saved to {path}")
    
    def load_preprocessor(self, path='./models/preprocessor.pkl'):
        """
        Load preprocessor objects
        """
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        
        logger.info(f"✓ Preprocessor loaded from {path}")


if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_training_data()
    df = loader.calculate_future_returns(df)
    df = loader.create_labels(df)
    
    # Get feature columns
    feature_cols = loader.get_feature_columns(df)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df, method='forward_fill')
    
    # Remove outliers (optional)
    if config.get('preprocessing.outlier_removal'):
        df = preprocessor.remove_outliers(df, feature_cols, 
                                         threshold=config.get('preprocessing.outlier_threshold'))
    
    # Prepare features and labels
    X, y, metadata = preprocessor.prepare_features_labels(df, feature_cols)
    
    print(f"\nPreprocessed data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nSample X:")
    print(X.head())
    print(f"\nSample y:")
    print(y.head())