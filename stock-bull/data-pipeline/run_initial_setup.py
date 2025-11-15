#!/usr/bin/env python3
"""
Initial setup script to collect all historical data
Run this once to populate the database
"""

import sys
from datetime import datetime, timedelta
import logging
from config.config import Config
from storage.database import DatabaseManager
from collectors.stock_price_collector import StockPriceCollector
from collectors.news_collector import NewsCollector
from collectors.fundamentals_collector import FundamentalsCollector
from processors.sentiment_analyzer import SentimentAnalyzer
from validators.data_validator import DataValidator
from processors.feature_generator import FeatureGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOGS_PATH}/initial_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def initial_setup():
    """
    Complete initial setup process
    """
    logger.info("\n" + "="*70)
    logger.info("STOCK BULL - INITIAL SETUP")
    logger.info("="*70 + "\n")
    
    print("This script will:")
    print("1. Create database tables")
    print("2. Collect 5-10 years of historical price data (50 stocks)")
    print("3. Collect index data (NIFTY 50)")
    print("4. Collect last 30 days of news articles")
    print("5. Analyze sentiment for all news")
    print("6. Collect fundamental data")
    print("7. Validate all data")
    print("8. Generate training dataset")
    print("\nEstimated time: 2-3 hours")
    print("="*70)
    
    confirm = input("\nDo you want to proceed? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Setup cancelled.")
        return
    
    try:
        # Step 1: Create database tables
        logger.info("\n[1/8] Creating database tables...")
        db = DatabaseManager()
        db.create_tables()
        
        # Step 2: Collect historical price data
        logger.info("\n[2/8] Collecting historical price data...")
        logger.info("This may take 60-90 minutes...")
        price_collector = StockPriceCollector()
        
        # Collect 5 years of data
        start_date = datetime.now() - timedelta(days=5*365)
        price_collector.collect_all_stocks(
            stocks_list=Config.NIFTY_50_STOCKS,
            start_date=start_date,
            end_date=datetime.now()
        )
        
        # Step 3: Collect index data
        logger.info("\n[3/8] Collecting NIFTY 50 index data...")
        price_collector.collect_index_data('NIFTY', start_date=start_date)
        
        # Step 4: Collect news articles
        logger.info("\n[4/8] Collecting news articles...")
        logger.info("Collecting last 30 days of news...")
        news_collector = NewsCollector()
        news_collector.collect_news_for_all_stocks(days_back=30)
        
        # Step 5: Analyze sentiment
        logger.info("\n[5/8] Analyzing sentiment...")
        logger.info("This may take 20-30 minutes...")
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_analyzer.analyze_news_articles(update_existing=False)
        
        # Step 6: Collect fundamentals
        logger.info("\n[6/8] Collecting fundamental data...")
        fundamentals_collector = FundamentalsCollector()
        fundamentals_collector.collect_all_fundamentals()
        
        # Step 7: Validate data
        logger.info("\n[7/8] Validating data...")
        validator = DataValidator()
        validator.validate_price_data()
        validator.validate_news_data()
        validator.generate_data_report()
        
        # Step 8: Generate training dataset
        logger.info("\n[8/8] Generating training dataset...")
        logger.info("This may take 30-45 minutes...")
        feature_generator = FeatureGenerator()
        
        # Generate dataset for last 3 years (enough for training)
        training_start = datetime.now() - timedelta(days=3*365)
        dataset = feature_generator.generate_training_dataset(
            stocks_list=Config.NIFTY_50_STOCKS[:10],  # Start with 10 stocks for testing
            start_date=training_start
        )
        
        if not dataset.empty:
            feature_generator.save_features(dataset, 'initial_training_dataset.csv')
        
        logger.info("\n" + "="*70)
        logger.info("✓ INITIAL SETUP COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Check the data quality report in logs")
        logger.info("2. Review the training dataset: data-pipeline/processed_data/initial_training_dataset.csv")
        logger.info("3. Start ML model training")
        logger.info("4. Set up daily scheduler: python pipeline_scheduler.py")
        
    except Exception as e:
        logger.error(f"\n✗ Setup failed with error: {e}")
        logger.error("Check the log file for details: data-pipeline/logs/initial_setup.log")
        raise


if __name__ == "__main__":
    initial_setup()