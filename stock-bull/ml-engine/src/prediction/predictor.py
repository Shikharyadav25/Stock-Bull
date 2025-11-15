import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
import sys
sys.path.append('../..')
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Make predictions on new stock data
    """
    
    def __init__(self, model_path='../models/saved_models/ensemble.pkl'):
        self.model = None
        self.preprocessor = None
        self.feature_selector = None
        self.class_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        
        self.load_components(model_path)
    
    def load_components(self, model_path):
        """Load model and preprocessing components"""
        logger.info("Loading model and components...")
        
        # Load model
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")
        
        # Load preprocessor
        preprocessor_path = Path(model_path).parent / 'preprocessor.pkl'
        if preprocessor_path.exists():
            preprocessor_data = joblib.load(preprocessor_path)
            self.preprocessor = preprocessor_data['scaler']
            self.feature_cols = preprocessor_data['feature_cols']
            logger.info(f"✓ Preprocessor loaded")
        else:
            logger.warning("Preprocessor not found, will use raw features")
        
        # Load feature selector (optional)
        selector_path = Path(model_path).parent / 'feature_selector.pkl'
        if selector_path.exists():
            selector_data = joblib.load(selector_path)
            self.selected_features = selector_data['selected_features']
            logger.info(f"✓ Feature selector loaded ({len(self.selected_features)} features)")
        else:
            self.selected_features = None
    
    def prepare_features(self, df):
        """
        Prepare features for prediction
        """
        # Select required features
        if self.selected_features:
            missing_features = [f for f in self.selected_features if f not in df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            available_features = [f for f in self.selected_features if f in df.columns]
            X = df[available_features].copy()
        else:
            X = df[self.feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        if self.preprocessor:
            X_scaled = self.preprocessor.transform(X.values)
        else:
            X_scaled = X.values
        
        return X_scaled
    
    def predict(self, df):
        """
        Predict stock classification
        """
        logger.info(f"Making predictions for {len(df)} records...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'symbol': df['symbol'].values if 'symbol' in df.columns else range(len(df)),
            'date': df['date'].values if 'date' in df.columns else None,
            'prediction': predictions,
            'prediction_label': [self.class_names[p] for p in predictions],
            'confidence': probabilities.max(axis=1)
        })
        
        # Add probability columns
        for i, class_name in enumerate(self.class_names):
            results[f'prob_{class_name}'] = probabilities[:, i]
        
        logger.info("✓ Predictions complete")
        
        return results
    
    def predict_single_stock(self, features_dict):
        """
        Predict for a single stock given feature dictionary
        """
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Predict
        results = self.predict(df)
        
        return {
            'prediction': int(results['prediction'].iloc[0]),
            'prediction_label': results['prediction_label'].iloc[0],
            'confidence': float(results['confidence'].iloc[0]),
            'probabilities': {
                class_name: float(results[f'prob_{class_name}'].iloc[0])
                for class_name in self.class_names
            }
        }


class StockScorer:
    """
    Calculate comprehensive stock scores
    """
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.weights = config.get('scoring.weights')
    
    def calculate_ml_score(self, prediction, confidence):
        """
        Calculate ML score (0-100) from prediction and confidence
        """
        # Map prediction class to base score
        class_scores = {
            0: 0,    # Strong Sell
            1: 25,   # Sell
            2: 50,   # Hold
            3: 75,   # Buy
            4: 100   # Strong Buy
        }
        
        base_score = class_scores.get(prediction, 50)
        
        # Adjust by confidence
        # If confidence is high, score stays near base
        # If confidence is low, score moves toward 50 (neutral)
        adjusted_score = base_score * confidence + 50 * (1 - confidence)
        
        return adjusted_score
    
    def calculate_technical_score(self, features):
        """
        Calculate technical indicator score (0-100)
        """
        score = 50  # Start neutral
        
        # RSI scoring
        if 'rsi' in features:
            rsi = features['rsi']
            if rsi < 30:
                score += 10  # Oversold - bullish
            elif rsi > 70:
                score -= 10  # Overbought - bearish
        
        # Moving average crossovers
        if 'sma_20' in features and 'sma_50' in features and 'close' in features:
            if features['close'] > features['sma_20'] > features['sma_50']:
                score += 10  # Bullish trend
            elif features['close'] < features['sma_20'] < features['sma_50']:
                score -= 10  # Bearish trend
        
        # MACD
        if 'macd' in features and 'macd_signal' in features:
            if features['macd'] > features['macd_signal']:
                score += 5  # Bullish
            else:
                score -= 5  # Bearish
        
        # Bollinger Bands
        if 'bb_lower' in features and 'bb_upper' in features and 'close' in features:
            if features['close'] < features['bb_lower']:
                score += 5  # Near lower band - potential bounce
            elif features['close'] > features['bb_upper']:
                score -= 5  # Near upper band - potential pullback
        
        # Volume
        if 'volume_ratio' in features:
            if features['volume_ratio'] > 1.5:
                score += 5  # High volume - strong conviction
        
        # Clip to 0-100
        return max(0, min(100, score))
    
    def calculate_sentiment_score(self, features):
        """
        Calculate sentiment score (0-100)
        """
        score = 50  # Start neutral
        
        if 'sentiment_mean' in features:
            # Sentiment ranges from -1 to 1
            sentiment = features['sentiment_mean']
            # Convert to 0-100 scale
            score = (sentiment + 1) * 50
        
        # Adjust for sentiment trend
        if 'sentiment_trend' in features:
            trend = features['sentiment_trend']
            if trend > 0.1:
                score += 10  # Improving sentiment
            elif trend < -0.1:
                score -= 10  # Declining sentiment
        
        # Clip to 0-100
        return max(0, min(100, score))
    
    def calculate_momentum_score(self, features):
        """
        Calculate momentum score (0-100)
        """
        score = 50
        
        # Recent returns
        if 'momentum_pct_5' in features:
            mom_5d = features['momentum_pct_5']
            if mom_5d > 5:
                score += 15
            elif mom_5d > 2:
                score += 10
            elif mom_5d < -5:
                score -= 15
            elif mom_5d < -2:
                score -= 10
        
        if 'momentum_pct_20' in features:
            mom_20d = features['momentum_pct_20']
            if mom_20d > 10:
                score += 10
            elif mom_20d < -10:
                score -= 10
        
        # Relative to market
        if 'relative_performance' in features:
            rel_perf = features['relative_performance']
            if rel_perf > 5:
                score += 10
            elif rel_perf < -5:
                score -= 10
        
        return max(0, min(100, score))
    
    def calculate_fundamental_score(self, features):
        """
        Calculate fundamental score (0-100)
        """
        score = 50
        
        # PE Ratio
        if 'pe_ratio' in features and features['pe_ratio'] > 0:
            pe = features['pe_ratio']
            if 10 <= pe <= 25:
                score += 10  # Reasonable PE
            elif pe > 40:
                score -= 10  # Overvalued
            elif pe < 5:
                score -= 5  # Too low - risky
        
        # PB Ratio
        if 'pb_ratio' in features and features['pb_ratio'] > 0:
            pb = features['pb_ratio']
            if pb < 1:
                score += 10  # Undervalued
            elif pb > 5:
                score -= 5  # Overvalued
        
        # Dividend Yield
        if 'dividend_yield' in features:
            div_yield = features['dividend_yield']
            if div_yield > 0.02:  # > 2%
                score += 5
        
        return max(0, min(100, score))
    
    def calculate_final_score(self, df, predictions):
        """
        Calculate final comprehensive stock score
        """
        logger.info("Calculating final scores...")
        
        scores = []
        
        for idx, row in df.iterrows():
            pred = predictions.loc[idx]
            
            # Individual component scores
            ml_score = self.calculate_ml_score(pred['prediction'], pred['confidence'])
            technical_score = self.calculate_technical_score(row)
            sentiment_score = self.calculate_sentiment_score(row)
            momentum_score = self.calculate_momentum_score(row)
            fundamental_score = self.calculate_fundamental_score(row)
            
            # Weighted final score
            final_score = (
                ml_score * self.weights['ml_score'] +
                technical_score * self.weights['technical_score'] +
                sentiment_score * self.weights['sentiment_score'] +
                momentum_score * self.weights['momentum_score'] +
                fundamental_score * self.weights['fundamental_score']
            )
            
            scores.append({
                'symbol': row.get('symbol', ''),
                'date': row.get('date', ''),
                'final_score': final_score,
                'ml_score': ml_score,
                'technical_score': technical_score,
                'sentiment_score': sentiment_score,
                'momentum_score': momentum_score,
                'fundamental_score': fundamental_score,
                'prediction': pred['prediction_label'],
                'confidence': pred['confidence']
            })
        
        scores_df = pd.DataFrame(scores)
        
        logger.info("✓ Score calculation complete")
        
        return scores_df


if __name__ == "__main__":
    # Test predictor
    import sys
    sys.path.append('../..')
    from src.data_preparation.data_loader import DataLoader
    
    # Load test data
    loader = DataLoader()
    df = loader.load_training_data()
    df = df.head(100)  # Test with 100 records
    
    # Initialize predictor
    predictor = StockPredictor('../models/saved_models/ensemble.pkl')
    
    # Make predictions
    predictions = predictor.predict(df)
    
    print("\nPrediction Results:")
    print(predictions.head(10))
    
    # Calculate scores
    scorer = StockScorer(predictor)
    scores = scorer.calculate_final_score(df, predictions)
    
    print("\nStock Scores:")
    print(scores.head(10))
    print(f"\nAverage final score: {scores['final_score'].mean():.2f}")