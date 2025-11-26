#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def quick_train():
    """Quick training for testing"""
    try:
        from src.data_preparation.data_loader import DataLoader
        from src.data_preparation.preprocessor import DataPreprocessor
        from src.models.random_forest_model import RandomForestModel
        from src.evaluation.evaluator import ModelEvaluator
        
        logger.info("Quick Training Mode - Testing Setup")
        
        # Load data
        logger.info("Loading data...")
        loader = DataLoader()
        df = loader.load_training_data()
        
        # Use only 6 months for quick test
        cutoff_date = datetime.now() - timedelta(days=180)
        df = df[df['date'] >= cutoff_date]
        
        # Use 10 stocks
        test_stocks = df['symbol'].unique()[:10]
        df = df[df['symbol'].isin(test_stocks)]
        
        logger.info(f"Using {len(df)} records from {len(test_stocks)} stocks")
        
        # Prepare data
        df = loader.calculate_future_returns(df, horizon_days=30)
        df = loader.create_labels(df)
        
        # Get feature columns - ONLY NUMERIC COLUMNS
        exclude_cols = ['symbol', 'date', 'future_return', 'label', 'label_name', 
                       'open', 'high', 'low', 'close', 'volume', 'created_at', 'updated_at']
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        logger.info(f"Selected {len(feature_cols)} numeric features")
        
        # Save metadata BEFORE preprocessing
        metadata_df = df[['symbol', 'date', 'close']].copy()
        
        # Preprocess
        preprocessor = DataPreprocessor()
        df = preprocessor.handle_missing_values(df)
        
        # Split data (simple 70/30 split)
        train_size = int(len(df) * 0.7)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Split metadata
        train_metadata = metadata_df[:train_size]
        test_metadata = metadata_df[train_size:]
        
        # Prepare features and labels (ONLY numeric features)
        X_train = train_df[feature_cols].copy()
        y_train = train_df['label'].copy()
        
        X_test = test_df[feature_cols].copy()
        y_test = test_df['label'].copy()
        
        # Ensure all data is numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        logger.info(f"Training set: {len(X_train)} samples, {len(feature_cols)} features")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Scale
        scaled = preprocessor.scale_features(X_train.values, X_test=X_test.values)
        X_train_scaled = scaled['train']
        X_test_scaled = scaled['test']
        
        # Train simple Random Forest
        logger.info("Training Random Forest...")
        model = RandomForestModel()
        model.train(X_train_scaled, y_train.values, X_test_scaled, y_test.values)
        
        # Evaluate
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Get actual classes present in test set
        unique_classes = np.unique(y_test.values)
        class_names_map = {0: 'Strong Sell', 1: 'Sell', 2: 'Hold', 3: 'Buy', 4: 'Strong Buy'}
        actual_class_names = [class_names_map[i] for i in unique_classes]
        
        # Create evaluator with actual classes
        evaluator = ModelEvaluator(class_names=actual_class_names)
        
        # Simple evaluation metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test.values, y_pred)
        precision = precision_score(y_test.values, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test.values, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test.values, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test.values, y_pred)
        
        logger.info("\n" + "="*60)
        logger.info("✓ QUICK TRAINING COMPLETE!")
        logger.info("="*60)
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"\nClasses in test set: {actual_class_names}")
        logger.info(f"Confusion Matrix:\n{cm}")
        logger.info("="*60)
        
        # Save model
        os.makedirs('../models/saved_models', exist_ok=True)
        model.save('../models/saved_models/quick_test_model.pkl')
        preprocessor.feature_cols = feature_cols
        preprocessor.save_preprocessor('../models/saved_models/preprocessor.pkl')
        
        logger.info("\n✓ Model saved to ../models/saved_models/quick_test_model.pkl")
        logger.info("✓ Preprocessor saved")
        
        logger.info("\n🎉 SUCCESS! Model training complete!")
        logger.info("\nNext step - Make predictions:")
        logger.info("  python scripts/predict.py")
        
        return {'accuracy': accuracy, 'f1_score': f1}
        
    except FileNotFoundError as e:
        logger.error(f"\n✗ Error: {e}")
        logger.error("Please ensure data pipeline has generated training data:")
        logger.error("  cd ../../stock-bull/data-pipeline")
        logger.error("  python run.py all")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    quick_train()