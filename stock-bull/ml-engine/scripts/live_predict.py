#!/usr/bin/env python3
"""
Live prediction with fresh news sentiment
"""
import sys
import os

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_engine_dir = os.path.dirname(script_dir)
stock_bull_dir = os.path.dirname(ml_engine_dir)
data_pipeline_dir = os.path.join(stock_bull_dir, 'data-pipeline')

sys.path.insert(0, data_pipeline_dir)

from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np

print("\n" + "="*70)
print("STOCK BULL - LIVE PREDICTIONS WITH FRESH NEWS")
print("="*70)

# Step 1: Collect fresh news
print("\n[1/4] Collecting latest news...")
from collectors.news_collector import NewsCollector

collector = NewsCollector()
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

total_articles = 0
for stock in stocks:
    print(f"  📰 {stock}...", end="", flush=True)
    count = collector.collect_news_google_rss(stock, max_articles=10)
    total_articles += count
    print(f" {count} articles")

print(f"  ✓ Total: {total_articles} articles collected")

# Step 2: Analyze sentiment
print("\n[2/4] Analyzing sentiment...")
from processors.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzed = analyzer.analyze_news_articles(update_existing=False)
print(f"  ✓ Analyzed {analyzed} new articles")

# Step 3: Get fresh sentiment data
print("\n[3/4] Calculating fresh sentiment scores...")
from storage.database import DatabaseManager, NewsArticle

db = DatabaseManager()
session = db.get_session()

# Get average sentiment for last 7 days
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

fresh_sentiment = {}
for stock in stocks:
    articles = session.query(NewsArticle).filter(
        NewsArticle.symbol == stock,
        NewsArticle.published_at >= start_date,
        NewsArticle.sentiment_score.isnot(None)
    ).all()
    
    if articles:
        scores = [a.sentiment_score for a in articles]
        avg_sentiment = sum(scores) / len(scores)
        fresh_sentiment[stock] = {
            'sentiment_mean': avg_sentiment,
            'news_count': len(articles),
            'latest_sentiment': scores[-1] if scores else 0
        }
        sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😐" if avg_sentiment > -0.1 else "😟"
        print(f"  {stock}: {sentiment_emoji} {len(articles)} articles, avg: {avg_sentiment:.2f}")
    else:
        fresh_sentiment[stock] = {'sentiment_mean': 0, 'news_count': 0, 'latest_sentiment': 0}
        print(f"  {stock}: ⚠️  No recent news")

session.close()

# Step 4: Make predictions with fresh data
print("\n[4/4] Making predictions with fresh sentiment...")

# Load model
model_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'quick_test_model.pkl')
preprocessor_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'preprocessor.pkl')

model = joblib.load(model_path)
preprocessor_data = joblib.load(preprocessor_path)
feature_cols = preprocessor_data['feature_cols']
scaler = preprocessor_data['scaler']

# Load latest market data
data_path = os.path.join(data_pipeline_dir, 'processed_data/complete_training_dataset.csv')
df = pd.read_csv(data_path)
latest_df = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)

# Class mapping
class_map = {0: 'Strong Sell', 1: 'Sell', 2: 'Hold', 3: 'Buy', 4: 'Strong Buy'}

print("\n" + "="*70)
print("LIVE PREDICTIONS WITH FRESH SENTIMENT (Last 7 Days)")
print("="*70)

results = []

for idx, row in latest_df.iterrows():
    symbol = row['symbol']
    
    # Get fresh sentiment
    fresh = fresh_sentiment.get(symbol, {})
    
    # Update sentiment features if they exist
    row_updated = row.copy()
    for col in feature_cols:
        if 'sentiment' in col.lower() and col in row.index:
            if 'mean' in col.lower():
                row_updated[col] = fresh.get('sentiment_mean', row[col])
            elif 'count' in col.lower():
                row_updated[col] = fresh.get('news_count', row[col])
    
    # Prepare features
    X = row_updated[feature_cols].values.reshape(1, -1)
    X_df = pd.DataFrame(X, columns=feature_cols)
    X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_scaled = scaler.transform(X_df)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    confidence = proba.max()
    
    # Store results
    results.append({
        'symbol': symbol,
        'prediction': class_map[pred],
        'confidence': confidence,
        'sentiment': fresh.get('sentiment_mean', 0),
        'news_count': fresh.get('news_count', 0),
        'latest_sentiment': fresh.get('latest_sentiment', 0),
        'price': row.get('close', 0)
    })

# Sort by confidence
results_df = pd.DataFrame(results).sort_values('confidence', ascending=False)

# Display
for _, r in results_df.iterrows():
    emoji = "✅" if r['prediction'] in ['Buy', 'Strong Buy'] else "⏸️" if r['prediction'] == 'Hold' else "❌"
    sentiment_emoji = "😊" if r['sentiment'] > 0.1 else "😐" if r['sentiment'] > -0.1 else "😟"
    
    print(f"\n{emoji} {r['symbol']}")
    print(f"   Price: ₹{r['price']:.2f}")
    print(f"   ML Prediction: {r['prediction']} (Confidence: {r['confidence']:.1%})")
    if r['news_count'] > 0:
        print(f"   Fresh Sentiment: {sentiment_emoji} {r['sentiment']:.2f} ({r['news_count']} articles)")
        latest_emoji = "📈" if r['latest_sentiment'] > 0.1 else "📉" if r['latest_sentiment'] < -0.1 else "➡️"
        print(f"   Latest News: {latest_emoji} {r['latest_sentiment']:.2f}")
    else:
        print(f"   Sentiment: ⚠️  No recent news")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

buy_stocks = results_df[results_df['prediction'].isin(['Buy', 'Strong Buy'])]
if len(buy_stocks) > 0:
    print(f"\n✅ BUY RECOMMENDATIONS:")
    for _, s in buy_stocks.iterrows():
        sent_text = f"(Sentiment: {s['sentiment']:.2f})" if s['news_count'] > 0 else "(No news)"
        print(f"   {s['symbol']}: {s['prediction']} - ML: {s['confidence']:.0%} {sent_text}")

hold_stocks = results_df[results_df['prediction'] == 'Hold']
if len(hold_stocks) > 0:
    print(f"\n⏸️  HOLD:")
    for _, s in hold_stocks.iterrows():
        print(f"   {s['symbol']}")

sell_stocks = results_df[results_df['prediction'].isin(['Sell', 'Strong Sell'])]
if len(sell_stocks) > 0:
    print(f"\n❌ AVOID:")
    for _, s in sell_stocks.iterrows():
        print(f"   {s['symbol']}")

print("\n" + "="*70)
print("✅ LIVE PREDICTION COMPLETE!")
print("="*70)
print(f"\n📊 Analysis based on:")
print(f"   • Fresh news from last 7 days")
print(f"   • AI sentiment analysis (FinBERT)")
print(f"   • 40+ technical indicators")
print(f"   • ML model (80.77% accuracy)")