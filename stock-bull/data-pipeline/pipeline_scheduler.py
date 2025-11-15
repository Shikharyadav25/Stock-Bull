import schedule
import time
from datetime import datetime
import logging
import sys
sys.path.append('.')
from config.config import Config
from collectors.stock_price_collector import StockPriceCollector
from collectors.news_collector import NewsCollector
from collectors.fundamentals_collector import FundamentalsCollector
from processors.sentiment_analyzer import SentimentAnalyzer
from validators.data_validator import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOGS_PATH}/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPipelineScheduler:
    """
    Automated scheduler for daily data collection
    """
    
    def __init__(self):
        self.price_collector = StockPriceCollector()
        self.news_collector = NewsCollector()
        self.fundamentals_collector = FundamentalsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.validator = DataValidator()
    
    def daily_update(self):
        """
        Daily data update job (run after market closes)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING DAILY DATA UPDATE")
        logger.info("="*60 + "\n")
        
        try:
            # 1. Update stock prices (last 7 days to ensure completeness)
            logger.info("Step 1: Updating stock prices...")
            self.price_collector.update_recent_data(days=7)
            
            # 2. Update index data
            logger.info("Step 2: Updating index data...")
            from datetime import timedelta
            self.price_collector.collect_index_data(
                'NIFTY',
                start_date=datetime.now()-timedelta(days=7)
            )
            
            # 3. Collect recent news
            logger.info("Step 3: Collecting recent news...")
            self.news_collector.collect_news_for_all_stocks(days_back=1)
            
            # 4. Analyze sentiment for new articles
            logger.info("Step 4: Analyzing sentiment...")
            self.sentiment_analyzer.analyze_news_articles(update_existing=False)
            
            # 5. Validate data
            logger.info("Step 5: Validating data...")
            self.validator.validate_price_data()
            
            logger.info("\n" + "="*60)
            logger.info("✓ DAILY UPDATE COMPLETED SUCCESSFULLY")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"✗ Error in daily update: {e}")
    
    def weekly_update(self):
        """
        Weekly data update job (comprehensive checks)
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING WEEKLY DATA UPDATE")
        logger.info("="*60 + "\n")
        
        try:
            # 1. Update fundamentals
            logger.info("Step 1: Updating fundamentals...")
            self.fundamentals_collector.collect_all_fundamentals()
            
            # 2. Collect more news
            logger.info("Step 2: Collecting weekly news...")
            self.news_collector.collect_news_for_all_stocks(days_back=7)
            
            # 3. Re-analyze all unanalyzed sentiment
            logger.info("Step 3: Analyzing sentiment...")
            self.sentiment_analyzer.analyze_news_articles(update_existing=False)
            
            # 4. Generate data quality report
            logger.info("Step 4: Generating data quality report...")
            self.validator.generate_data_report()
            
            logger.info("\n" + "="*60)
            logger.info("✓ WEEKLY UPDATE COMPLETED SUCCESSFULLY")
            logger.info("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"✗ Error in weekly update: {e}")
    
    def run_scheduler(self):
        """
        Run the scheduler
        """
        logger.info("Starting Data Pipeline Scheduler...")
        
        # Schedule daily update (run at 6:00 PM IST, after market closes at 3:30 PM)
        schedule.every().day.at("18:00").do(self.daily_update)
        
        # Schedule weekly update (run on Sunday at 10:00 AM)
        schedule.every().sunday.at("10:00").do(self.weekly_update)
        
        logger.info("Scheduler configured:")
        logger.info("  - Daily updates: 6:00 PM IST")
        logger.info("  - Weekly updates: Sunday 10:00 AM IST")
        logger.info("\nPress Ctrl+C to stop\n")
        
        # Run continuously
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


if __name__ == "__main__":
    print("\nStock Bull - Data Pipeline Scheduler")
    print("="*50)
    print("1. Run scheduler (continuous)")
    print("2. Run daily update now")
    print("3. Run weekly update now")
    print("="*50)
    
    choice = input("\nEnter your choice (1-3): ")
    
    scheduler = DataPipelineScheduler()
    
    if choice == '1':
        scheduler.run_scheduler()
    
    elif choice == '2':
        scheduler.daily_update()
    
    elif choice == '3':
        scheduler.weekly_update()
    
    else:
        print("Invalid choice!")