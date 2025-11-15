from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import sys
sys.path.append('..')
from config.config import Config

Base = declarative_base()

# Define Database Models

class Stock(Base):
    __tablename__ = 'stocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(200))
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    isin = Column(String(20))
    listing_date = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DailyPrice(Base):
    __tablename__ = 'daily_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    adj_close = Column(Float)
    change_percent = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        {'extend_existing': True}
    )


class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    content = Column(Text)
    source = Column(String(100))
    author = Column(String(200))
    url = Column(Text, unique=True)
    published_at = Column(DateTime, index=True)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive/negative/neutral
    created_at = Column(DateTime, default=datetime.utcnow)


class IndexData(Base):
    __tablename__ = 'index_data'
    
    id = Column(Integer, primary_key=True)
    index_name = Column(String(50), nullable=False, index=True)  # NIFTY50, SENSEX
    date = Column(Date, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class StockFundamentals(Base):
    __tablename__ = 'stock_fundamentals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    dividend_yield = Column(Float)
    eps = Column(Float)
    book_value = Column(Float)
    face_value = Column(Float)
    market_cap = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataCollectionLog(Base):
    __tablename__ = 'data_collection_logs'
    
    id = Column(Integer, primary_key=True)
    collection_type = Column(String(50))  # prices, news, fundamentals
    symbol = Column(String(20))
    start_date = Column(Date)
    end_date = Column(Date)
    records_collected = Column(Integer)
    status = Column(String(20))  # success, failed, partial
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database Connection Manager
class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        print("✓ All tables created successfully")
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
        print("✗ All tables dropped")


# Initialize database
if __name__ == "__main__":
    db = DatabaseManager()
    db.create_tables()