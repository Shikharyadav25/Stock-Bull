import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from nsepy import get_history
from sqlalchemy.exc import IntegrityError
import sys
sys.path.append('..')
from config.config import Config
from storage.database import DatabaseManager, DailyPrice, Stock, IndexData, DataCollectionLog

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOGS_PATH}/price_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockPriceCollector:
    def __init__(self):
        self.db = DatabaseManager()
        self.session = self.db.get_session()
    
    def collect_stock_prices_nsepy(self, symbol, start_date, end_date):
        """
        Collect stock prices using NSEPy (recommended for NSE stocks)
        """
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Get data from NSEPy
            df = get_history(
                symbol=symbol,
                start=start_date,
                end=end_date
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return 0
            
            # Process and store data
            records_added = 0
            for index, row in df.iterrows():
                try:
                    price_record = DailyPrice(
                        symbol=symbol,
                        date=index.date(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume']),
                        adj_close=float(row['Close'])
                    )
                    
                    # Calculate change percent - CONVERT TO PYTHON FLOAT
                    if row['Open'] > 0:
                        change_pct = ((row['Close'] - row['Open']) / row['Open']) * 100
                        price_record.change_percent = float(change_pct)
                    else:
                        price_record.change_percent = 0.0
                    
                    self.session.merge(price_record)
                    records_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row for {symbol}: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"✓ Added/Updated {records_added} records for {symbol}")
            
            # Log collection
            self._log_collection('prices', symbol, start_date, end_date, records_added, 'success')
            
            return records_added
            
        except Exception as e:
            logger.error(f"✗ Error fetching data for {symbol}: {e}")
            self._log_collection('prices', symbol, start_date, end_date, 0, 'failed', str(e))
            self.session.rollback()
            return 0
    
    def collect_stock_prices_yfinance(self, symbol, start_date, end_date):
        """
        Collect stock prices using yfinance (backup method)
        """
        try:
            # Add .NS suffix for NSE stocks
            ticker_symbol = f"{symbol}.NS"
            logger.info(f"Fetching data for {ticker_symbol} from {start_date} to {end_date}")
            
            # Download data
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data found for {ticker_symbol}")
                return 0
            
            # Process and store data
            records_added = 0
            for index, row in df.iterrows():
                try:
                    price_record = DailyPrice(
                        symbol=symbol,
                        date=index.date(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume']),
                        adj_close=float(row['Close'])
                    )
                    
                    # Calculate change percent - CONVERT TO PYTHON FLOAT
                    if row['Open'] > 0:
                        change_pct = ((row['Close'] - row['Open']) / row['Open']) * 100
                        price_record.change_percent = float(change_pct)
                    else:
                        price_record.change_percent = 0.0
                    
                    self.session.merge(price_record)
                    records_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row for {symbol}: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"✓ Added/Updated {records_added} records for {symbol}")
            
            self._log_collection('prices', symbol, start_date, end_date, records_added, 'success')
            
            return records_added
            
        except Exception as e:
            logger.error(f"✗ Error fetching data for {symbol}: {e}")
            self._log_collection('prices', symbol, start_date, end_date, 0, 'failed', str(e))
            self.session.rollback()
            return 0
    
    def collect_index_data(self, index_name='NIFTY', start_date=None, end_date=None):
        """
        Collect index data (NIFTY 50, SENSEX)
        """
        try:
            if start_date is None:
                start_date = Config.DATA_START_DATE
            if end_date is None:
                end_date = datetime.now()
            
            logger.info(f"Fetching {index_name} index data")
            
            # Get NIFTY 50 data
            df = get_history(
                symbol=index_name,
                start=start_date,
                end=end_date,
                index=True
            )
            
            if df.empty:
                logger.warning(f"No data found for {index_name}")
                return 0
            
            records_added = 0
            for index, row in df.iterrows():
                try:
                    index_record = IndexData(
                        index_name=index_name,
                        date=index.date(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row.get('Volume', 0))
                    )
                    
                    self.session.merge(index_record)
                    records_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing index row: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"✓ Added/Updated {records_added} records for {index_name}")
            return records_added
            
        except Exception as e:
            logger.error(f"✗ Error fetching index data: {e}")
            self.session.rollback()
            return 0
    
    def collect_all_stocks(self, stocks_list=None, start_date=None, end_date=None):
        """
        Collect price data for all stocks in the list
        """
        if stocks_list is None:
            stocks_list = Config.NIFTY_50_STOCKS
        
        if start_date is None:
            start_date = Config.DATA_START_DATE
        if end_date is None:
            end_date = datetime.now()
        
        logger.info(f"Starting bulk collection for {len(stocks_list)} stocks")
        
        total_records = 0
        success_count = 0
        
        for i, symbol in enumerate(stocks_list, 1):
            logger.info(f"Progress: {i}/{len(stocks_list)} - {symbol}")
            
            # Try NSEPy first
            records = self.collect_stock_prices_nsepy(symbol, start_date, end_date)
            
            # If NSEPy fails, try yfinance
            if records == 0:
                logger.info(f"Trying yfinance for {symbol}")
                records = self.collect_stock_prices_yfinance(symbol, start_date, end_date)
            
            if records > 0:
                success_count += 1
                total_records += records
            
            # Rate limiting
            time.sleep(Config.API_DELAY)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Collection Complete!")
        logger.info(f"Successful: {success_count}/{len(stocks_list)} stocks")
        logger.info(f"Total records: {total_records}")
        logger.info(f"{'='*60}\n")
        
        return total_records
    
    def update_recent_data(self, days=7):
        """
        Update data for the last N days (for daily updates)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Updating recent data from {start_date.date()} to {end_date.date()}")
        
        return self.collect_all_stocks(
            stocks_list=Config.NIFTY_50_STOCKS,
            start_date=start_date,
            end_date=end_date
        )
    
    def _log_collection(self, collection_type, symbol, start_date, end_date, records, status, error=None):
        """Log data collection activity"""
        try:
            log_entry = DataCollectionLog(
                collection_type=collection_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                records_collected=records,
                status=status,
                error_message=error
            )
            self.session.add(log_entry)
            self.session.commit()
        except Exception as e:
            logger.error(f"Error logging collection: {e}")
            self.session.rollback()
    
    def __del__(self):
        """Cleanup"""
        if self.session:
            self.session.close()


# CLI Interface
if __name__ == "__main__":
    collector = StockPriceCollector()
    
    print("\nStock Bull - Price Data Collector")
    print("="*50)
    print("1. Collect historical data (all stocks, 5-10 years)")
    print("2. Update recent data (last 7 days)")
    print("3. Collect data for specific stock")
    print("4. Collect index data (NIFTY 50)")
    print("="*50)
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        print("\nStarting historical data collection...")
        print("This may take 30-60 minutes for 50 stocks")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            collector.collect_all_stocks()
            collector.collect_index_data('NIFTY')
    
    elif choice == '2':
        print("\nUpdating recent data...")
        collector.update_recent_data(days=7)
        collector.collect_index_data('NIFTY', start_date=datetime.now()-timedelta(days=7))
    
    elif choice == '3':
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").upper()
        start_date = input("Enter start date (YYYY-MM-DD) or press Enter for 5 years: ")
        
        if start_date:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_date = datetime.now() - timedelta(days=5*365)
        
        collector.collect_stock_prices_nsepy(symbol, start_date, datetime.now())
    
    elif choice == '4':
        print("\nCollecting NIFTY 50 index data...")
        collector.collect_index_data('NIFTY')
    
    else:
        print("Invalid choice!")