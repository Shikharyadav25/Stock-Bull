# Stock Bull - Data Pipeline Documentation

## Overview
The data pipeline collects, processes, and validates stock market data from multiple sources for the Stock Bull platform.

## Architecture
```
┌─────────────────┐
│  Data Sources   │
├─────────────────┤
│ • NSEPy         │
│ • yfinance      │
│ • NewsAPI       │
│ • Google RSS    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Collectors    │
├─────────────────┤
│ • Prices        │
│ • News          │
│ • Fundamentals  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL    │
│    Database     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Processors    │
├─────────────────┤
│ • Tech Indic.   │
│ • Sentiment     │
│ • Features      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Training    │
│    Dataset      │
└─────────────────┘
```

## Installation

### 1. Prerequisites
```bash
# Python 3.8+
python --version

# PostgreSQL 12+
psql --version
```

### 2. Setup Virtual Environment
```bash
cd stock-bull/data-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install TA-Lib (for technical indicators)

**Ubuntu/Debian:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Windows:**
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
```

### 4. Configure Database
```bash
# Create PostgreSQL database
createdb stockbull

# Or using psql
psql postgres
CREATE DATABASE stockbull;
CREATE USER your_username WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE stockbull TO your_username;
\q
```

### 5. Configure Environment
Create `.env` file:
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stockbull
DB_USER=your_username
DB_PASSWORD=your_password

# API Keys (optional but recommended)
NEWS_API_KEY=your_newsapi_key_here

# Settings
DATA_START_DATE=2019-01-01
```

Get free NewsAPI key: https://newsapi.org/register

## Initial Setup

### Run Complete Setup (Recommended)
```bash
python run_initial_setup.py
```

This will:
- Create all database tables
- Collect 5 years of historical data
- Collect news and analyze sentiment
- Generate training dataset

**Time Required:** 2-3 hours

### Manual Step-by-Step Setup

**Step 1: Create Tables**
```bash
python storage/database.py
```

**Step 2: Collect Price Data**
```bash
python collectors/stock_price_collector.py
# Choose option 1: Collect historical data
```

**Step 3: Collect News Data**
```bash
python collectors/news_collector.py
# Choose option 1: Collect news for all stocks
```

**Step 4: Analyze Sentiment**
```bash
python processors/sentiment_analyzer.py
# Choose option 1: Analyze all unanalyzed articles
```

**Step 5: Collect Fundamentals**
```bash
python collectors/fundamentals_collector.py
```

**Step 6: Validate Data**
```bash
python validators/data_validator.py
# Choose option 3: Generate data quality report
```

**Step 7: Generate Features**
```bash
python processors/feature_generator.py
# Choose option 2: Generate complete training dataset
```

## Daily Operations

### Automated Scheduler
```bash
# Run continuous scheduler (recommended for production)
python pipeline_scheduler.py
# Choose option 1: Run scheduler (continuous)
```

The scheduler will automatically:
- Update prices daily at 6:00 PM IST
- Collect news daily
- Run weekly comprehensive updates on Sundays

### Manual Updates
```bash
# Update recent data manually
python collectors/stock_price_collector.py
# Choose option 2: Update recent data

# Collect latest news
python collectors/news_collector.py
# Choose option 3: Update recent news
```

## Data Pipeline Components

### 1. Stock Price Collector
**File:** `collectors/stock_price_collector.py`

**Features:**
- Collects OHLCV data from NSE/BSE
- Uses NSEPy (primary) and yfinance (backup)
- Handles rate limiting and retries
- Logs all collection activities

**Usage:**
```python
from collectors.stock_price_collector import StockPriceCollector

collector = StockPriceCollector()
collector.collect_stock_prices_nsepy('RELIANCE', start_date, end_date)
collector.collect_all_stocks()
```

### 2. News Collector
**File:** `collectors/news_collector.py`

**Features:**
- Collects from NewsAPI (100 req/day free)
- Scrapes Google News RSS (unlimited, free)
- Maps stock symbols to company names
- Deduplicates articles

**Usage:**
```python
from collectors.news_collector import NewsCollector

collector = NewsCollector()
collector.collect_news_newsapi('RELIANCE', days_back=30)
collector.collect_news_google_rss('RELIANCE')
```

### 3. Sentiment Analyzer
**File:** `processors/sentiment_analyzer.py`

**Features:**
- Uses FinBERT (financial sentiment model)
- Fallback to rule-based sentiment
- Aggregates sentiment by day/week/month
- Calculates sentiment trends

**Usage:**
```python
from processors.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.analyze_news_articles(symbol='RELIANCE')
summary = analyzer.get_sentiment_aggregates('RELIANCE', days=30)
```

### 4. Technical Indicators
**File:** `processors/technical_indicators.py`

**Features:**
- 50+ technical indicators
- SMA, EMA, RSI, MACD, Bollinger Bands
- Volume indicators, momentum indicators
- Support/resistance levels

**Usage:**
```python
from processors.technical_indicators import TechnicalIndicators

df = TechnicalIndicators.calculate_all_indicators(price_df)
```

### 5. Feature Generator
**File:** `processors/feature_generator.py`

**Features:**
- Combines price, sentiment, fundamental data
- Generates 70+ features per stock-day
- Creates ML-ready datasets
- Handles missing data

**Usage:**
```python
from processors.feature_generator import FeatureGenerator

generator = FeatureGenerator()
features = generator.generate_complete_features('RELIANCE')
dataset = generator.generate_training_dataset()
```

## Database Schema

### Tables

**stocks**
- Stock metadata (symbol, name, sector, market cap)

**daily_prices**
- OHLCV data for each stock
- Calculated change percentages

**news_articles**
- News headlines and content
- Sentiment scores and labels
- Source and publication date

**index_data**
- Market index data (NIFTY 50, SENSEX)

**stock_fundamentals**
- PE ratio, PB ratio, dividend yield
- EPS, market cap

**data_collection_logs**
- Tracks all data collection activities
- Error logging and debugging

## Data Quality & Validation

### Validation Checks
1. **Price Data:**
   - No null values in critical fields
   - OHLC logic (High ≥ Low, etc.)
   - No negative prices
   - Suspicious price jumps (>50%)
   - Missing trading days

2. **News Data:**
   - No duplicate URLs
   - Valid publication dates
   - Sentiment scores in valid range

3. **Completeness:**
   - Data coverage per stock
   - Date range validation
   - Records per stock

### Run Validation
```bash
python validators/data_validator.py
```

## Troubleshooting

### Common Issues

**1. Database Connection Error**
```
Error: could not connect to server
```
Solution:
- Verify PostgreSQL is running: `sudo service postgresql status`
- Check `.env` credentials
- Test connection: `psql -U your_username -d stockbull`

**2. NSEPy Returns Empty Data**
```
No data found for symbol
```
Solution:
- NSE servers may be temporarily down
- Fallback will use yfinance automatically
- Check symbol name is correct

**3. NewsAPI Rate Limit**
```
Error 429: Rate limit exceeded
```
Solution:
- Free tier has 100 requests/day
- Use Google RSS fallback (unlimited)
- Or upgrade NewsAPI plan

**4. FinBERT Model Loading Error**
```
Error loading sentiment model
```
Solution:
- Ensure transformers and torch are installed
- Check internet connection (first-time download)
- Fallback will use rule-based sentiment

**5. Missing TA-Lib**
```
ImportError: TA-Lib not installed
```
Solution:
- Follow TA-Lib installation steps above
- Or comment out TA-Lib indicators in technical_indicators.py

### Logs Location
All logs are stored in: `data-pipeline/logs/`
- `price_collector.log`
- `news_collector.log`
- `sentiment_analyzer.log`
- `scheduler.log`

## Performance Optimization

### Speed Tips
1. **Parallel Collection:** Modify collectors to use threading
2. **Batch Processing:** Process multiple stocks simultaneously
3. **Caching:** Cache frequently accessed data
4. **Incremental Updates:** Only update changed data

### Resource Management
- PostgreSQL max connections: 100
- API rate limits: Respect delays
- Memory: ~2-4GB for full dataset
- Disk: ~5-10GB for 5 years of data

## Data Pipeline Outputs

### Generated Files
```
data-pipeline/
├── raw_data/
│   └── (temporary storage)
├── processed_data/
│   ├── initial_training_dataset.csv
│   ├── {SYMBOL}_features.csv
│   └── complete_training_dataset.csv
└── logs/
    ├── price_collector.log
    ├── news_collector.log
    └── sentiment_analyzer.log
```

### Feature Dataset Columns
The training dataset includes 70+ features:
- **Price Features:** OHLCV, returns, volatility
- **Technical Indicators:** SMA, EMA, RSI, MACD, Bollinger Bands
- **Sentiment Features:** Daily/weekly/monthly sentiment scores
- **Fundamental Features:** PE, PB, dividend yield, market cap
- **Market Features:** Index returns, market volatility
- **Derived Features:** Relative performance, momentum

## Next Steps

After data pipeline setup:
1. ✅ Data collection complete
2. 📊 Start ML model training (see ml-engine documentation)
3. 🚀 Build backend API
4. 💻 Develop frontend application

## Support

For issues or questions:
- Check logs in `data-pipeline/logs/`
- Review error messages
- Verify API keys and database credentials

## License
MIT License - See LICENSE file