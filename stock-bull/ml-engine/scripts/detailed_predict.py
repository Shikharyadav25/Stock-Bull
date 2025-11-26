#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
import numpy as np

# Get absolute paths
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_engine_dir = os.path.dirname(script_dir)
stock_bull_dir = os.path.dirname(ml_engine_dir)

# Load model
print("Loading model...")
model_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'quick_test_model.pkl')
preprocessor_path = os.path.join(stock_bull_dir, 'models', 'saved_models', 'preprocessor.pkl')

model = joblib.load(model_path)
preprocessor_data = joblib.load(preprocessor_path)
feature_cols = preprocessor_data['feature_cols']
scaler = preprocessor_data['scaler']

# Load data
data_path = os.path.join(stock_bull_dir, 'data-pipeline', 'processed_data', 'complete_training_dataset.csv')
df = pd.read_csv(data_path)

# Get latest for each stock
latest_df = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)

# Check for sentiment columns
sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]

print("\n" + "="*80)
print("STOCK BULL - DETAILED PREDICTIONS WITH SENTIMENT ANALYSIS")
print("="*80)
print(f"\nAnalyzing {len(latest_df)} stocks...")
print(f"Latest date: {latest_df['date'].max()}")
print(f"Sentiment features found: {len(sentiment_cols)}")
print("="*80)

# Class mapping
class_map = {0: 'Strong Sell', 1: 'Sell', 2: 'Hold', 3: 'Buy', 4: 'Strong Buy'}

# Make predictions
results = []

for idx, row in latest_df.iterrows():
    symbol = row['symbol']
    
    try:
        # Prepare features
        X = row[feature_cols].values.reshape(1, -1)
        X_df = pd.DataFrame(X, columns=feature_cols)
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_scaled = scaler.transform(X_df)
        
        # Predict
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        confidence = proba.max()
        
        # Get sentiment data if available
        sentiment_mean = row.get('sentiment_mean', 0)
        sentiment_trend = row.get('sentiment_trend', 0)
        news_count = row.get('news_count', 0)
        
        # Handle NaN values
        if pd.isna(sentiment_mean):
            sentiment_mean = 0
        if pd.isna(sentiment_trend):
            sentiment_trend = 0
        if pd.isna(news_count):
            news_count = 0
        
        # Get technical indicators
        rsi = row.get('rsi', 0)
        macd = row.get('macd', 0)
        
        if pd.isna(rsi):
            rsi = 0
        if pd.isna(macd):
            macd = 0
        
        results.append({
            'symbol': symbol,
            'prediction': class_map.get(pred, 'Unknown'),
            'confidence': confidence,
            'close_price': row.get('close', 0),
            'sentiment_score': sentiment_mean,
            'sentiment_trend': sentiment_trend,
            'news_count': int(news_count),
            'rsi': rsi,
            'macd': macd
        })
    except Exception as e:
        print(f"Error predicting {symbol}: {e}")

# Sort by confidence
results_df = pd.DataFrame(results).sort_values('confidence', ascending=False)

# Display detailed results
print("\n" + "="*80)
print("DETAILED ANALYSIS")
print("="*80)

for _, row in results_df.iterrows():
    emoji = "✅" if row['prediction'] in ['Buy', 'Strong Buy'] else "⏸️" if row['prediction'] == 'Hold' else "❌"
    
    print(f"\n{emoji} {row['symbol']}")
    print(f"   Price: ₹{row['close_price']:.2f}")
    print(f"   ML Prediction: {row['prediction']} (Confidence: {row['confidence']:.1%})")
    
    # Sentiment Analysis
    if row['sentiment_score'] != 0:
        sentiment_emoji = "😊" if row['sentiment_score'] > 0.1 else "😐" if row['sentiment_score'] > -0.1 else "😟"
        trend_emoji = "📈" if row['sentiment_trend'] > 0 else "📉" if row['sentiment_trend'] < 0 else "➡️"
        
        print(f"   News Sentiment: {sentiment_emoji} {row['sentiment_score']:.2f} {trend_emoji}")
        print(f"   Articles analyzed: {row['news_count']}")
    else:
        print(f"   News Sentiment: ⚠️  No news data")
    
    # Technical Indicators
    print(f"   Technical: RSI={row['rsi']:.1f}, MACD={row['macd']:.2f}")
    print("-" * 80)

# Summary
print("\n" + "="*80)
print("RECOMMENDATION SUMMARY")
print("="*80)

buy_stocks = results_df[results_df['prediction'].isin(['Buy', 'Strong Buy'])]
if len(buy_stocks) > 0:
    print(f"\n✅ BUY RECOMMENDATIONS:")
    for _, stock in buy_stocks.iterrows():
        print(f"   {stock['symbol']: <10} → {stock['prediction']} "
              f"(ML: {stock['confidence']:.0%}, Sentiment: {stock['sentiment_score']:.2f})")

hold_stocks = results_df[results_df['prediction'] == 'Hold']
if len(hold_stocks) > 0:
    print(f"\n⏸️  HOLD:")
    for _, stock in hold_stocks.iterrows():
        print(f"   {stock['symbol']}")

print("\n" + "="*80)
print("🎉 ANALYSIS COMPLETE!")
print("="*80)