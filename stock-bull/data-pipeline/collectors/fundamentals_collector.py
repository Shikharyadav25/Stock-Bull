import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import logging
import sys
sys.path.append('..')
from config.config import Config
from storage.database import DatabaseManager, Stock, StockFundamentals

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOGS_PATH}/fundamentals_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FundamentalsCollector:
    def __init__(self):
        self.db = DatabaseManager()
        self.session = self.db.get_session()
    
    def collect_stock_info(self, symbol):
        """
        Collect basic stock information and fundamentals
        """
        try:
            ticker_symbol = f"{symbol}.NS"
            logger.info(f"Fetching fundamentals for {ticker_symbol}")
            
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            
            # Update/Create stock record
            stock = self.session.query(Stock).filter_by(symbol=symbol).first()
            
            if not stock:
                stock = Stock(symbol=symbol)
            
            # Update stock information
            stock.company_name = info.get('longName', symbol)
            stock.sector = info.get('sector', 'Unknown')
            stock.industry = info.get('industry', 'Unknown')
            stock.market_cap = info.get('marketCap')
            
            self.session.merge(stock)
            
            # Create fundamentals record
            fundamental = StockFundamentals(
                symbol=symbol,
                date=datetime.now().date(),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                eps=info.get('trailingEps'),
                book_value=info.get('bookValue'),
                market_cap=info.get('marketCap')
            )
            
            self.session.add(fundamental)
            self.session.commit()
            
            logger.info(f"✓ Updated fundamentals for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error fetching fundamentals for {symbol}: {e}")
            self.session.rollback()
            return False
    
    def collect_all_fundamentals(self, stocks_list=None):
        """
        Collect fundamentals for all stocks
        """
        if stocks_list is None:
            stocks_list = Config.NIFTY_50_STOCKS
        
        logger.info(f"Starting fundamentals collection for {len(stocks_list)} stocks")
        
        success_count = 0
        
        for i, symbol in enumerate(stocks_list, 1):
            logger.info(f"Progress: {i}/{len(stocks_list)} - {symbol}")
            
            if self.collect_stock_info(symbol):
                success_count += 1
            
            time.sleep(Config.API_DELAY)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Fundamentals Collection Complete!")
        logger.info(f"Successful: {success_count}/{len(stocks_list)} stocks")
        logger.info(f"{'='*60}\n")
        
        return success_count
    
    def __del__(self):
        """Cleanup"""
        if self.session:
            self.session.close()


if __name__ == "__main__":
    collector = FundamentalsCollector()
    
    print("\nStock Bull - Fundamentals Data Collector")
    print("="*50)
    print("Collecting fundamental data for all stocks...")
    
    confirm = input("Continue? (yes/no): ")
    if confirm.lower() == 'yes':
        collector.collect_all_fundamentals()