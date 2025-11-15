import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
sys.path.append('..')
from config.config import Config
from storage.database import DatabaseManager, DailyPrice, NewsArticle, IndexData, StockFundamentals
from processors.technical_indicators import TechnicalIndicators
from processors.sentiment_analyzer import SentimentAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureGenerator:
    """
    Generate complete feature set for ML models
    """
    
    def __init__(self):
        self.db = DatabaseManager()
        self.session = self.db.get_session()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def get_price_features(self, symbol, start_date=None, end_date=None):
        """
        Get price data and calculate technical indicators
        """
        logger.info(f"Generating price features for {symbol}...")
        
        # Query price data
        query = self.session.query(DailyPrice).filter_by(symbol=symbol)
        
        if start_date:
            query = query.filter(DailyPrice.date >= start_date)
        if end_date:
            query = query.filter(DailyPrice.date <= end_date)
        
        query = query.order_by(DailyPrice.date)
        
        df = pd.read_sql(query.statement, self.session.bind)
        
        if df.empty:
            logger.warning(f"No price data found for {symbol}")
            return pd.DataFrame()
        
        # Calculate technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        return df
    
    def get_sentiment_features(self, symbol, start_date=None, end_date=None):
        """
        Get sentiment features from news data
        """
        logger.info(f"Generating sentiment features for {symbol}...")
        
        # Query news articles
        query = self.session.query(NewsArticle).filter_by(symbol=symbol)
        
        if start_date:
            query = query.filter(NewsArticle.published_at >= start_date)
        if end_date:
            query = query.filter(NewsArticle.published_at <= end_date)
        
        articles = query.all()
        
        if not articles:
            logger.warning(f"No news data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        news_df = pd.DataFrame([{
            'date': a.published_at.date(),
            'sentiment_score': a.sentiment_score or 0,
            'sentiment_label': a.sentiment_label or 'neutral'
        } for a in articles])
        
        # Aggregate sentiment by day
        daily_sentiment = news_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                                   'news_count', 'sentiment_min', 'sentiment_max']
        
        # Fill missing values
        daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)
        
        # Calculate rolling sentiment metrics
        daily_sentiment = daily_sentiment.sort_values('date')
        daily_sentiment['sentiment_7d_avg'] = daily_sentiment['sentiment_mean'].rolling(7, min_periods=1).mean()
        daily_sentiment['sentiment_30d_avg'] = daily_sentiment['sentiment_mean'].rolling(30, min_periods=1).mean()
        daily_sentiment['sentiment_trend'] = daily_sentiment['sentiment_7d_avg'] - daily_sentiment['sentiment_30d_avg']
        
        return daily_sentiment
    
    def get_fundamental_features(self, symbol):
        """
        Get fundamental data features
        """
        logger.info(f"Generating fundamental features for {symbol}...")
        
        # Get latest fundamentals
        fundamental = self.session.query(StockFundamentals).filter_by(
            symbol=symbol
        ).order_by(StockFundamentals.date.desc()).first()
        
        if not fundamental:
            logger.warning(f"No fundamental data found for {symbol}")
            return {}
        
        return {
            'pe_ratio': fundamental.pe_ratio or 0,
            'pb_ratio': fundamental.pb_ratio or 0,
            'dividend_yield': fundamental.dividend_yield or 0,
            'eps': fundamental.eps or 0,
            'market_cap': fundamental.market_cap or 0
        }
    
    def get_market_features(self, start_date=None, end_date=None):
        """
        Get market index features (NIFTY 50)
        """
        logger.info("Generating market features...")
        
        # Query index data
        query = self.session.query(IndexData).filter_by(index_name='NIFTY')
        
        if start_date:
            query = query.filter(IndexData.date >= start_date)
        if end_date:
            query = query.filter(IndexData.date <= end_date)
        
        query = query.order_by(IndexData.date)
        
        df = pd.read_sql(query.statement, self.session.bind)
        
        if df.empty:
            logger.warning("No market index data found")
            return pd.DataFrame()
        
        # Calculate market returns
        df['market_return'] = df['close'].pct_change() * 100
        df['market_return_5d'] = df['close'].pct_change(5) * 100
        df['market_return_20d'] = df['close'].pct_change(20) * 100
        
        # Market volatility
        df['market_volatility'] = df['market_return'].rolling(20).std()
        
        return df[['date', 'market_return', 'market_return_5d', 
                  'market_return_20d', 'market_volatility']]
    
    def generate_complete_features(self, symbol, start_date=None, end_date=None):
        """
        Generate complete feature set by combining all feature types
        """
        logger.info(f"Generating complete feature set for {symbol}...")
        
        if start_date is None:
            start_date = Config.DATA_START_DATE
        if end_date is None:
            end_date = datetime.now()
        
        # Get price features (base dataset)
        price_df = self.get_price_features(symbol, start_date, end_date)
        
        if price_df.empty:
            logger.error(f"Cannot generate features without price data for {symbol}")
            return pd.DataFrame()
        
        # Get sentiment features
        sentiment_df = self.get_sentiment_features(symbol, start_date, end_date)
        
        # Get market features
        market_df = self.get_market_features(start_date, end_date)
        
        # Merge all features
        # Start with price data
        features_df = price_df.copy()
        
        # Merge sentiment data
        if not sentiment_df.empty:
            features_df = features_df.merge(
                sentiment_df,
                on='date',
                how='left'
            )
            # Fill missing sentiment with neutral (0)
            sentiment_cols = ['sentiment_mean', 'sentiment_7d_avg', 'sentiment_30d_avg', 'sentiment_trend']
            for col in sentiment_cols:
                if col in features_df.columns:
                    features_df[col] = features_df[col].fillna(0)
            if 'news_count' in features_df.columns:
                features_df['news_count'] = features_df['news_count'].fillna(0)
        
        # Merge market data
        if not market_df.empty:
            features_df = features_df.merge(
                market_df,
                on='date',
                how='left'
            )
        
        # Add fundamental features (constant for all dates)
        fundamentals = self.get_fundamental_features(symbol)
        for key, value in fundamentals.items():
            features_df[key] = value
        
        # Add derived features
        features_df['symbol'] = symbol
        
        # Calculate relative performance vs market
        if 'market_return' in features_df.columns and 'change_percent' in features_df.columns:
            features_df['relative_performance'] = features_df['change_percent'] - features_df['market_return']
        
        # Drop rows with too many NaN values (typically first 200 rows due to indicators)
        features_df = features_df.dropna(thresh=len(features_df.columns) * 0.7)
        
        logger.info(f"✓ Generated {len(features_df.columns)} features for {len(features_df)} days")
        
        return features_df
    
    def generate_training_dataset(self, stocks_list=None, start_date=None, end_date=None):
        """
        Generate complete training dataset for all stocks
        """
        if stocks_list is None:
            stocks_list = Config.NIFTY_50_STOCKS
        
        logger.info(f"Generating training dataset for {len(stocks_list)} stocks...")
        
        all_features = []
        
        for i, symbol in enumerate(stocks_list, 1):
            logger.info(f"Processing {i}/{len(stocks_list)}: {symbol}")
            
            features = self.generate_complete_features(symbol, start_date, end_date)
            
            if not features.empty:
                all_features.append(features)
        
        if not all_features:
            logger.error("No features generated!")
            return pd.DataFrame()
        
        # Combine all stocks
        final_df = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Dataset Generated!")
        logger.info(f"Total records: {len(final_df):,}")
        logger.info(f"Stocks: {final_df['symbol'].nunique()}")
        logger.info(f"Features: {len(final_df.columns)}")
        logger.info(f"Date range: {final_df['date'].min()} to {final_df['date'].max()}")
        logger.info(f"{'='*60}\n")
        
        return final_df
    
    def save_features(self, df, filename='features_dataset.csv'):
        """
        Save features to CSV file
        """
        filepath = f"{Config.PROCESSED_DATA_PATH}/{filename}"
        df.to_csv(filepath, index=False)
        logger.info(f"✓ Features saved to {filepath}")
    
    def __del__(self):
        """Cleanup"""
        if self.session:
            self.session.close()


# CLI Interface
if __name__ == "__main__":
    generator = FeatureGenerator()
    
    print("\nStock Bull - Feature Generator")
    print("="*50)
    print("1. Generate features for single stock")
    print("2. Generate complete training dataset")
    print("3. Generate features for last 30 days (quick test)")
    print("="*50)
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").upper()
        features = generator.generate_complete_features(symbol)
        
        if not features.empty:
            print(f"\nGenerated {len(features.columns)} features:")
            print(features.tail())
            
            save = input("\nSave to CSV? (yes/no): ")
            if save.lower() == 'yes':
                generator.save_features(features, f'{symbol}_features.csv')
    
    elif choice == '2':
        print("\nGenerating complete training dataset...")
        print("This will take 30-45 minutes for 50 stocks")
        confirm = input("Continue? (yes/no): ")
        
        if confirm.lower() == 'yes':
            dataset = generator.generate_training_dataset()
            
            if not dataset.empty:
                generator.save_features(dataset, 'complete_training_dataset.csv')
                
                print("\nDataset Summary:")
                print(f"Shape: {dataset.shape}")
                print(f"\nSample data:")
                print(dataset.head())
                print(f"\nFeature list:")
                print(dataset.columns.tolist())
    
    elif choice == '3':
        print("\nQuick test: Generating features for last 30 days...")
        test_stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        
        start_date = datetime.now() - timedelta(days=30)
        dataset = generator.generate_training_dataset(
            stocks_list=test_stocks,
            start_date=start_date
        )
        
        if not dataset.empty:
            print(f"\nGenerated {len(dataset)} records")
            print(dataset.tail(10))
    
    else:
        print("Invalid choice!")