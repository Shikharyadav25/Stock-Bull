# Stock Bull Data Pipeline - Quick Start Guide

## 🚀 Get Started in 30 Minutes (Test Mode)

This guide will help you set up a working data pipeline quickly with a small dataset for testing.

### Step 1: Install Dependencies (5 minutes)
```bash
# Clone or navigate to project
cd stock-bull/data-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install pandas numpy yfinance nsepy newsapi-python requests beautifulsoup4 sqlalchemy psycopg2-binary python-dotenv schedule transformers torch
```

### Step 2: Setup Database (5 minutes)
```bash
# Install PostgreSQL (if not installed)
# Ubuntu: sudo apt-get install postgresql
# macOS: brew install postgresql
# Windows: Download from postgresql.org

# Create database
createdb stockbull

# Or using psql
psql postgres -c "CREATE DATABASE stockbull;"
```

### Step 3: Configure Environment (2 minutes)

Create `.env` file in `data-pipeline/` directory:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stockbull
DB_USER=postgres
DB_PASSWORD=your_password

# Optional: Get free key at newsapi.org
NEWS_API_KEY=your_key_here
```

### Step 4: Run Quick Setup (15 minutes)
```bash
# Create tables
python storage/database.py

# Collect data for 5 stocks, last 30 days (quick test)
python << EOF
from datetime import datetime, timedelta
from collectors.stock_price_collector import StockPriceCollector
from collectors.news_collector import NewsCollector
from processors.sentiment_analyzer import SentimentAnalyzer

# Test stocks
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
start_date = datetime.now() - timedelta(days=30)

# Collect prices
print("Collecting prices...")
pc = StockPriceCollector()
for stock in stocks:
    pc.collect_stock_prices_yfinance(stock, start_date, datetime.now())

# Collect news
print("Collecting news...")
nc = NewsCollector()
for stock in stocks:
    nc.collect_news_google_rss(stock, max_articles=10)

# Analyze sentiment
print("Analyzing sentiment...")
sa = SentimentAnalyzer()
sa.analyze_news_articles()

print("✓ Quick setup complete!")
EOF
```

### Step 5: Verify Data (3 minutes)
```bash
python validators/data_validator.py
# Choose option 3: Generate data quality report
```

You should see:
- ✅ Price data for 5 stocks
- ✅ News articles with sentiment
- ✅ Date range: last 30 days

### Step 6: Generate Features
```bash
python processors/feature_generator.py
# Choose option 3: Generate features for last 30 days
```

## 🎉 Success!

You now have:
- Working database with sample data
- Price data + technical indicators
- News data + sentiment analysis
- Features ready for ML training

## Next Steps

### For College Project (Quick):
1. Use the 30-day dataset for model training
2. Train a simple classifier
3. Build basic frontend
4. Demo ready in 1-2 weeks!

### For Full Production:
1. Run `python run_initial_setup.py` for complete data
2. Set up automated scheduler
3. Expand to more stocks
4. Deploy to cloud

## Common Issues

**Database connection error?**
- Check PostgreSQL is running: `sudo service postgresql start`
- Verify `.env` credentials

**No news collected?**
- Google RSS is free and works without API key
- NewsAPI needs free registration

**Import errors?**
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

## Test Your Setup
```python
# Run this to verify everything works
python << EOF
import sys
sys.path.append('.')
from storage.database import DatabaseManager
from sqlalchemy import func
from storage.database import DailyPrice, NewsArticle

db = DatabaseManager()
session = db.get_session()

price_count = session.query(func.count(DailyPrice.id)).scalar()
news_count = session.query(func.count(NewsArticle.id)).scalar()

print(f"✓ Price records: {price_count}")
print(f"✓ News articles: {news_count}")
print(f"✓ Setup successful!" if price_count > 0 else "✗ No data found")

session.close()
EOF
```

Good luck with your project! 🚀
```

## Summary: Complete Data Pipeline Structure
```
stock-bull/
└── data-pipeline/
    ├── config/
    │   └── config.py                    # Configuration settings
    ├── storage/
    │   └── database.py                   # Database models & connection
    ├── collectors/
    │   ├── stock_price_collector.py      # Price data collection
    │   ├── news_collector.py             # News collection
    │   └── fundamentals_collector.py     # Fundamental data
    ├── processors/
    │   ├── technical_indicators.py       # Technical analysis
    │   ├── sentiment_analyzer.py         # Sentiment analysis
    │   └── feature_generator.py          # Feature engineering
    ├── validators/
    │   └── data_validator.py             # Data quality checks
    ├── pipeline_scheduler.py             # Automated scheduler
    ├── run_initial_setup.py              # One-time setup script
    ├── requirements.txt                  # Python dependencies
    ├── .env                              # Environment variables
    ├── README_DATA_PIPELINE.md           # Full documentation
    ├── QUICKSTART.md                     # Quick start guide
    ├── raw_data/                         # Temporary storage
    ├── processed_data/                   # Generated datasets
    └── logs/                             # Log files