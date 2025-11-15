from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
sys.path.append('..')
from storage.database import DatabaseManager, NewsArticle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyze sentiment of news articles using FinBERT
    """
    
    def __init__(self, model_name='ProsusAI/finbert'):
        """
        Initialize sentiment analyzer with FinBERT model
        """
        try:
            logger.info(f"Loading sentiment model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("✓ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to simple rule-based sentiment")
            self.sentiment_pipeline = None
        
        self.db = DatabaseManager()
        self.session = self.db.get_session()
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a single text
        Returns: (sentiment_label, sentiment_score)
        """
        if not text or len(text.strip()) == 0:
            return 'neutral', 0.0
        
        try:
            if self.sentiment_pipeline:
                # Use FinBERT for sentiment analysis
                # Truncate text to avoid token limit
                text = text[:512]
                
                result = self.sentiment_pipeline(text)[0]
                label = result['label'].lower()
                score = result['score']
                
                # Convert to -1 to 1 scale
                if label == 'positive':
                    sentiment_score = score
                elif label == 'negative':
                    sentiment_score = -score
                else:  # neutral
                    sentiment_score = 0.0
                
                return label, sentiment_score
            else:
                # Simple rule-based fallback
                return self._simple_sentiment(text)
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 'neutral', 0.0
    
    def _simple_sentiment(self, text):
        """
        Simple rule-based sentiment analysis (fallback)
        """
        text = text.lower()
        
        positive_words = [
            'profit', 'gain', 'growth', 'up', 'rise', 'surge', 'positive',
            'strong', 'bullish', 'high', 'beat', 'exceed', 'success',
            'improve', 'boost', 'rally', 'soar', 'jump', 'win', 'record'
        ]
        
        negative_words = [
            'loss', 'decline', 'down', 'fall', 'drop', 'negative', 'weak',
            'bearish', 'low', 'miss', 'fail', 'concern', 'worry', 'risk',
            'crash', 'plunge', 'tumble', 'slump', 'crisis', 'warning'
        ]
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 'neutral', 0.0
        
        score = (pos_count - neg_count) / total
        
        if score > 0.2:
            return 'positive', score
        elif score < -0.2:
            return 'negative', score
        else:
            return 'neutral', score
    
    def analyze_news_articles(self, symbol=None, update_existing=False):
        """
        Analyze sentiment for news articles in database
        """
        logger.info(f"Analyzing sentiment for articles{' for ' + symbol if symbol else ''}...")
        
        # Query articles
        query = self.session.query(NewsArticle)
        
        if symbol:
            query = query.filter_by(symbol=symbol)
        
        if not update_existing:
            query = query.filter(NewsArticle.sentiment_score.is_(None))
        
        articles = query.all()
        
        if not articles:
            logger.warning("No articles to analyze")
            return 0
        
        logger.info(f"Analyzing {len(articles)} articles...")
        
        analyzed_count = 0
        
        for i, article in enumerate(articles, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(articles)}")
            
            try:
                # Combine title and description for analysis
                text = f"{article.title} {article.description or ''}"
                
                label, score = self.analyze_text(text)
                
                # Update article
                article.sentiment_label = label
                article.sentiment_score = score
                
                analyzed_count += 1
                
            except Exception as e:
                logger.error(f"Error analyzing article {article.id}: {e}")
                continue
        
        # Commit changes
        self.session.commit()
        
        logger.info(f"✓ Analyzed {analyzed_count} articles")
        
        return analyzed_count
    
    def get_sentiment_aggregates(self, symbol, days=30):
        """
        Get aggregated sentiment metrics for a stock
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        articles = self.session.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.published_at >= start_date,
            NewsArticle.sentiment_score.isnot(None)
        ).all()
        
        if not articles:
            return {
                'avg_sentiment': 0.0,
                'sentiment_trend': 'neutral',
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        scores = [a.sentiment_score for a in articles]
        labels = [a.sentiment_label for a in articles]
        
        avg_sentiment = np.mean(scores)
        
        # Calculate trend (comparing first half vs second half)
        mid_point = len(scores) // 2
        if mid_point > 0:
            first_half_avg = np.mean(scores[:mid_point])
            second_half_avg = np.mean(scores[mid_point:])
            
            if second_half_avg > first_half_avg + 0.1:
                trend = 'improving'
            elif second_half_avg < first_half_avg - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'avg_sentiment': float(avg_sentiment),
            'sentiment_trend': trend,
            'article_count': len(articles),
            'positive_count': labels.count('positive'),
            'negative_count': labels.count('negative'),
            'neutral_count': labels.count('neutral'),
            'sentiment_volatility': float(np.std(scores))
        }
    
    def get_daily_sentiment(self, symbol, days=30):
        """
        Get daily aggregated sentiment scores
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        articles = self.session.query(NewsArticle).filter(
            NewsArticle.symbol == symbol,
            NewsArticle.published_at >= start_date,
            NewsArticle.sentiment_score.isnot(None)
        ).all()
        
        if not articles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = [{
            'date': a.published_at.date(),
            'sentiment_score': a.sentiment_score
        } for a in articles]
        
        df = pd.DataFrame(data)
        
        # Aggregate by day
        daily_sentiment = df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).reset_index()
        
        daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'article_count']
        
        return daily_sentiment
    
    def __del__(self):
        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
        except:
            pass


# CLI Interface
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    print("\nStock Bull - Sentiment Analyzer")
    print("="*50)
    print("1. Analyze all unanalyzed articles")
    print("2. Analyze articles for specific stock")
    print("3. Re-analyze all articles")
    print("4. Get sentiment summary for stock")
    print("="*50)
    
    choice = input("\nEnter your choice (1-4): ")
    
    if choice == '1':
        print("\nAnalyzing all unanalyzed articles...")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            analyzer.analyze_news_articles(update_existing=False)
    
    elif choice == '2':
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").upper()
        analyzer.analyze_news_articles(symbol=symbol, update_existing=False)
    
    elif choice == '3':
        print("\nRe-analyzing all articles...")
        print("WARNING: This will overwrite existing sentiment scores")
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() == 'yes':
            analyzer.analyze_news_articles(update_existing=True)
    
    elif choice == '4':
        symbol = input("Enter stock symbol (e.g., RELIANCE): ").upper()
        days = int(input("Enter days to analyze (default 30): ") or "30")
        
        summary = analyzer.get_sentiment_aggregates(symbol, days)
        
        print(f"\nSentiment Summary for {symbol} (last {days} days):")
        print("="*50)
        print(f"Average Sentiment: {summary['avg_sentiment']:.3f}")
        print(f"Sentiment Trend: {summary['sentiment_trend']}")
        print(f"Total Articles: {summary['article_count']}")
        print(f"Positive: {summary['positive_count']}")
        print(f"Negative: {summary['negative_count']}")
        print(f"Neutral: {summary['neutral_count']}")
        print(f"Volatility: {summary['sentiment_volatility']:.3f}")
        print("="*50)
    
    else:
        print("Invalid choice!")