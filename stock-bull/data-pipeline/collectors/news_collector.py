from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import sys
sys.path.append('..')
from config.config import Config
from storage.database import DatabaseManager, NewsArticle, DataCollectionLog

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{Config.LOGS_PATH}/news_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NewsCollector:
    def __init__(self):
        self.db = DatabaseManager()
        self.session = self.db.get_session()
        self.newsapi = NewsApiClient(api_key=Config.NEWS_API_KEY) if Config.NEWS_API_KEY else None
    
    def get_company_keywords(self, symbol):
        """
        Map stock symbols to company names and keywords for better search
        """
        company_map = {
            'RELIANCE': ['Reliance Industries', 'RIL', 'Mukesh Ambani'],
            'TCS': ['Tata Consultancy Services', 'TCS'],
            'HDFCBANK': ['HDFC Bank', 'HDFC'],
            'INFY': ['Infosys', 'Infosys Technologies'],
            'ICICIBANK': ['ICICI Bank', 'ICICI'],
            'ITC': ['ITC Limited', 'ITC'],
            'SBIN': ['State Bank of India', 'SBI'],
            'BHARTIARTL': ['Bharti Airtel', 'Airtel'],
            'WIPRO': ['Wipro Limited', 'Wipro'],
            'LT': ['Larsen Toubro', 'L&T'],
            # Add more mappings as needed
        }
        
        return company_map.get(symbol, [symbol])
    
    def collect_news_newsapi(self, symbol, days_back=30):
        """
        Collect news using NewsAPI (100 requests/day on free tier)
        """
        if not self.newsapi:
            logger.error("NewsAPI key not configured")
            return 0
        
        try:
            keywords = self.get_company_keywords(symbol)
            query = ' OR '.join(keywords)
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            logger.info(f"Fetching news for {symbol} with query: {query}")
            
            # Fetch articles
            articles = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            if not articles or 'articles' not in articles:
                logger.warning(f"No articles found for {symbol}")
                return 0
            
            records_added = 0
            for article in articles['articles']:
                try:
                    # Check if article already exists
                    existing = self.session.query(NewsArticle).filter_by(url=article['url']).first()
                    if existing:
                        continue
                    
                    news_record = NewsArticle(
                        symbol=symbol,
                        title=article['title'],
                        description=article.get('description', ''),
                        content=article.get('content', ''),
                        source=article['source']['name'],
                        author=article.get('author', ''),
                        url=article['url'],
                        published_at=datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    )
                    
                    self.session.add(news_record)
                    records_added += 1
                    
                except Exception as e:
                    logger.error(f"Error processing article: {e}")
                    continue
            
            self.session.commit()
            logger.info(f"✓ Added {records_added} news articles for {symbol}")
            
            self._log_collection('news', symbol, from_date, to_date, records_added, 'success')
            
            return records_added
            
        except Exception as e:
            logger.error(f"✗ Error fetching news for {symbol}: {e}")
            self._log_collection('news', symbol, from_date, to_date, 0, 'failed', str(e))
            self.session.rollback()
            return 0
    
    def collect_news_google_rss(self, symbol, max_articles=20):
        try:
            keywords = self.get_company_keywords(symbol)
            articles_collected = 0
        
            for keyword in keywords[:2]:  # Use first 2 keywords
            # Google News RSS URL - SIMPLIFIED (removed when:1m which was causing issues)
                search_term = keyword.replace(' ', '+')
                rss_url = f"https://news.google.com/rss/search?q={search_term}&hl=en-IN&gl=IN&ceid=IN:en"
            
                logger.info(f"Fetching Google News RSS for {keyword}")
            
                try:
                    headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(rss_url, headers=headers, timeout=10)
                
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch RSS for {keyword}")
                        continue
                
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')[:max_articles]
                
                    logger.info(f"Found {len(items)} articles for {keyword}")
                
                    for item in items:
                        try:
                            title = item.find('title').text
                            link = item.find('link').text
                            pub_date_str = item.find('pubDate').text
                            source = item.find('source').text if item.find('source') else 'Google News'
                        
                        # Parse date
                            pub_date = datetime.strptime(pub_date_str, '%a, %d %b %Y %H:%M:%S %Z')
                        
                        # Check if already exists
                            existing = self.session.query(NewsArticle).filter_by(url=link).first()
                            if existing:
                                continue
                        
                            news_record = NewsArticle(
                                symbol=symbol,
                                title=title,
                                description='',
                                content='',
                                source=source,
                                author='',
                                url=link,
                                published_at=pub_date
                            )
                        
                            self.session.add(news_record)
                            articles_collected += 1
                        
                        except Exception as e:
                            logger.error(f"Error parsing RSS item: {e}")
                            continue
                
                    time.sleep(1)  # Rate limiting
                
                except Exception as e:
                    logger.error(f"Error fetching RSS for {keyword}: {e}")
                    continue
        
            self.session.commit()
            logger.info(f"✓ Collected {articles_collected} articles from Google RSS for {symbol}")
        
            return articles_collected
        
        except Exception as e:
            logger.error(f"✗ Error in Google RSS collection for {symbol}: {e}")
            self.session.rollback()
            return 0
    
    def collect_news_for_all_stocks(self, stocks_list=None, days_back=30):
        """
        Collect news for all stocks in the list
        """
        if stocks_list is None:
            stocks_list = Config.NIFTY_50_STOCKS
        
        logger.info(f"Starting news collection for {len(stocks_list)} stocks")
        
        total_articles = 0
        
        for i, symbol in enumerate(stocks_list, 1):
            logger.info(f"Progress: {i}/{len(stocks_list)} - {symbol}")
            
            # Try NewsAPI if available
            if self.newsapi:
                articles = self.collect_news_newsapi(symbol, days_back)
            else:
                articles = 0
            
            # Always use Google RSS as well (it's free)
            articles += self.collect_news_google_rss(symbol, max_articles=15)
            
            total_articles += articles
            
            # Rate limiting
            time.sleep(Config.API_DELAY)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"News Collection Complete!")
        logger.info(f"Total articles: {total_articles}")
        logger.info(f"{'='*60}\n")
        
        return total_articles
    
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
    
    def __del__(self):
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
        except:
            pass  # Silently ignore cleanup errors


# CLI Interface
if __name__ == "__main__":
    collector = NewsCollector()
    
    print("\nStock Bull - News Data Collector")
    print("="*50)
    print("1. Collect news for all stocks (last 30 days)")
    print("2. Collect news for specific stock")
    print("3. Update recent news (last 7 days)")
    print("="*50)
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        print("\nStarting news collection for all stocks...")
        print("This may take 20-30 minutes")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            collector.collect_news_for_all_stocks(days_back=30)
    
    elif choice == '2':
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").upper()
        days = int(input("Enter days back (default 30): ") or "30")
        if collector.newsapi:
            collector.collect_news_newsapi(symbol, days_back=days)
        else:
            collector.collect_news_google_rss(symbol, max_articles=30)
    
    elif choice == '3':
        print("\nUpdating recent news...")
        collector.collect_news_for_all_stocks(days_back=7)
    
    else:
        print("Invalid choice!")