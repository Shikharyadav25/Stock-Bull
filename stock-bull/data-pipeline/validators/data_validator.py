import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
sys.path.append('..')
from config.config import Config
from storage.database import DatabaseManager, DailyPrice, NewsArticle, Stock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self):
        self.db = DatabaseManager()
        self.session = self.db.get_session()
    
    def validate_price_data(self, symbol=None):
        """
        Validate price data for completeness and quality
        """
        logger.info("Validating price data...")
        
        # Query price data
        query = self.session.query(DailyPrice)
        if symbol:
            query = query.filter_by(symbol=symbol)
        
        df = pd.read_sql(query.statement, self.session.bind)
        
        if df.empty:
            logger.error("No price data found!")
            return False
        
        issues = []
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for OHLC logic (High >= Low, Close between High and Low)
        if (df['high'] < df['low']).any():
            issues.append("High < Low anomaly detected")
        
        # Check for suspicious price jumps (>50% in a day)
        df_sorted = df.sort_values(['symbol', 'date'])
        df_sorted['pct_change'] = df_sorted.groupby('symbol')['close'].pct_change() * 100
        suspicious = df_sorted[abs(df_sorted['pct_change']) > 50]
        if not suspicious.empty:
            issues.append(f"Suspicious price jumps detected: {len(suspicious)} records")
        
        # Check for data gaps (missing dates)
        if symbol:
            dates = pd.to_datetime(df['date']).sort_values()
            date_range = pd.date_range(start=dates.min(), end=dates.max(), freq='D')
            missing_dates = date_range.difference(dates)
            # Filter out weekends
            missing_dates = [d for d in missing_dates if d.weekday() < 5]
            if len(missing_dates) > 0:
                issues.append(f"Missing {len(missing_dates)} trading days")
        
        # Print validation results
        if issues:
            logger.warning(f"Validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            return False
        else:
            logger.info("✓ Price data validation passed!")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"  Stocks: {df['symbol'].nunique()}")
            return True
    
    def validate_news_data(self, symbol=None):
        """
        Validate news data
        """
        logger.info("Validating news data...")
        
        query = self.session.query(NewsArticle)
        if symbol:
            query = query.filter_by(symbol=symbol)
        
        df = pd.read_sql(query.statement, self.session.bind)
        
        if df.empty:
            logger.warning("No news data found!")
            return False
        
        issues = []
        
        # Check for null titles
        if df['title'].isnull().any():
            issues.append(f"Null titles found: {df['title'].isnull().sum()}")
        
        # Check for duplicate URLs
        duplicates = df[df.duplicated('url', keep=False)]
        if not duplicates.empty:
            issues.append(f"Duplicate URLs found: {len(duplicates)}")
        
        # Check date range
        date_range = (df['published_at'].max() - df['published_at'].min()).days
        if date_range < 7:
            issues.append(f"Limited date range: only {date_range} days")
        
        # Print validation results
        if issues:
            logger.warning(f"Validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ News data validation passed!")
        
        logger.info(f"  Total articles: {len(df)}")
        logger.info(f"  Date range: {df['published_at'].min()} to {df['published_at'].max()}")
        logger.info(f"  Stocks covered: {df['symbol'].nunique()}")
        
        return len(issues) == 0
    
    def generate_data_report(self):
        """
        Generate comprehensive data quality report
        """
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)
        
        # Price data summary
        prices = pd.read_sql(
            self.session.query(DailyPrice).statement,
            self.session.bind
        )
        logger.info(f"\nPrice Data:")
        logger.info(f"  Total records: {len(prices):,}")
        logger.info(f"  Stocks: {prices['symbol'].nunique()}")
        logger.info(f"  Date range: {prices['date'].min()} to {prices['date'].max()}")
        logger.info(f"  Avg records per stock: {len(prices) / prices['symbol'].nunique():.0f}")
        
        # News data summary
        news = pd.read_sql(
            self.session.query(NewsArticle).statement,
            self.session.bind
        )
        if not news.empty:
            logger.info(f"\nNews Data:")
            logger.info(f"  Total articles: {len(news):,}")
            logger.info(f"  Stocks covered: {news['symbol'].nunique()}")
            logger.info(f"  Date range: {news['published_at'].min()} to {news['published_at'].max()}")
            logger.info(f"  Avg articles per stock: {len(news) / news['symbol'].nunique():.0f}")
        
        # Stock info summary
        stocks = pd.read_sql(
            self.session.query(Stock).statement,
            self.session.bind
        )
        if not stocks.empty:
            logger.info(f"\nStock Info:")
            logger.info(f"  Total stocks: {len(stocks)}")
            logger.info(f"  Sectors: {stocks['sector'].nunique()}")
        
        logger.info("\n" + "="*60 + "\n")
    
    def __del__(self):
        """Cleanup"""
        if self.session:
            self.session.close()


if __name__ == "__main__":
    validator = DataValidator()
    
    print("\nStock Bull - Data Validator")
    print("="*50)
    print("1. Validate all data")
    print("2. Validate specific stock")
    print("3. Generate data quality report")
    print("="*50)
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        validator.validate_price_data()
        validator.validate_news_data()
    
    elif choice == '2':
        symbol = input("Enter stock symbol: ").upper()
        validator.validate_price_data(symbol)
        validator.validate_news_data(symbol)
    
    elif choice == '3':
        validator.generate_data_report()
    
    else:
        print("Invalid choice!")