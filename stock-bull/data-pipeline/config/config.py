import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

class Config:
    # Database Configuration
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'stockbull')
    DB_USER = os.getenv('DB_USER', 'shikharyadav')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # API Keys
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    # Data Settings
    DATA_START_DATE = datetime.strptime(os.getenv('DATA_START_DATE', '2019-01-01'), '%Y-%m-%d')
    DATA_END_DATE = datetime.now()
    
    # Stock Lists
    NIFTY_50_STOCKS = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'BAJFINANCE',
        'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
        'TITAN', 'SUNPHARMA', 'ULTRACEMCO', 'NESTLEIND', 'WIPRO'
    ]
    
    # Get the actual directory we're in
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Paths - Use absolute paths
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'raw_data')
    PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'processed_data')
    LOGS_PATH = os.path.join(BASE_DIR, 'logs')
    
    # Rate Limiting
    API_DELAY = 0.5
    MAX_RETRIES = 3

# Create directories if they don't exist
os.makedirs(Config.RAW_DATA_PATH, exist_ok=True)
os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(Config.LOGS_PATH, exist_ok=True)