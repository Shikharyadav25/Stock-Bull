#!/usr/bin/env python3
"""
Runner script that handles imports correctly
"""
import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def setup_database():
    """Setup database tables"""
    print("Setting up database...")
    from storage.database import DatabaseManager
    
    db = DatabaseManager()
    db.create_tables()
    print("✓ Database tables created")

def test_data_collection():
    """Test data collection"""
    print("\nTesting data collection...")
    from datetime import datetime, timedelta
    from collectors.stock_price_collector import StockPriceCollector
    
    collector = StockPriceCollector()
    start_date = datetime.now() - timedelta(days=30)
    
    print("Collecting RELIANCE data...")
    result = collector.collect_stock_prices_yfinance('RELIANCE', start_date, datetime.now())
    print(f"✓ Collected {result} records")

def collect_test_data():
    """Collect test data for 10 stocks"""
    print("\nCollecting test data for training...")
    from datetime import datetime, timedelta
    from collectors.stock_price_collector import StockPriceCollector
    from collectors.fundamentals_collector import FundamentalsCollector
    
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE']
    start_date = datetime.now() - timedelta(days=90)
    
    # Collect prices
    print("\n1. Collecting price data...")
    price_collector = StockPriceCollector()
    for i, symbol in enumerate(stocks, 1):
        print(f"  [{i}/{len(stocks)}] {symbol}...")
        price_collector.collect_stock_prices_yfinance(symbol, start_date, datetime.now())
    
    # Collect fundamentals
    print("\n2. Collecting fundamentals...")
    fund_collector = FundamentalsCollector()
    for i, symbol in enumerate(stocks, 1):
        print(f"  [{i}/{len(stocks)}] {symbol}...")
        fund_collector.collect_stock_info(symbol)
    
    print("\n✓ Data collection complete!")

def generate_features():
    """Generate features for ML"""
    print("\nGenerating features...")
    from datetime import datetime, timedelta
    from processors.feature_generator import FeatureGenerator
    
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE']
    start_date = datetime.now() - timedelta(days=90)
    
    generator = FeatureGenerator()
    dataset = generator.generate_training_dataset(
        stocks_list=stocks,
        start_date=start_date
    )
    
    if not dataset.empty:
        generator.save_features(dataset, 'complete_training_dataset.csv')
        print(f"✓ Features saved: {len(dataset)} records")
        return True
    else:
        print("✗ Failed to generate features")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Pipeline Runner')
    parser.add_argument('command', choices=['setup', 'test', 'collect', 'features', 'all'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'setup':
            setup_database()
        elif args.command == 'test':
            test_data_collection()
        elif args.command == 'collect':
            collect_test_data()
        elif args.command == 'features':
            generate_features()
        elif args.command == 'all':
            setup_database()
            collect_test_data()
            generate_features()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)