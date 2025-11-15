#!/usr/bin/env python3
"""
Complete training pipeline for Stock Bull ML models
"""

import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import joblib

from src.data_preparation.data_loader import DataLoader
from src.data_preparation.preprocessor import DataPreprocessor
from src.feature_engineering.feature_selector import FeatureSelector
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ensemble_model import EnsembleModel
from src.evaluation.evaluator import ModelEvaluator
from config.config_loader import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Complete ML training pipeline
    """
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.feature_selector = FeatureSelector(
            method=config.get('feature_engineering.selection_method'),
            max_features=config.get('feature_engineering.max_features')
        )
        self.evaluator = ModelEvaluator()
        
        self.models = {}
        self.results = {}
        
        # Create output directories
        Path('../models/saved_models').mkdir(parents=True, exist_ok=True)
        Path('../logs').mkdir(parents=True, exist_ok=True)
        Path('../data/processed').mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare data"""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*70)
        
        # Load training data
        self.df = self.data_loader.load_training_data()
        
        # Calculate future returns
        prediction_horizon = config.get('data.prediction_horizon', 30)
        self.df = self.data_loader.calculate_future_returns(self.df, horizon_days=prediction_horizon)
        
        # Create labels
        self.df = self.data_loader.create_labels(self.df)
        
        # Get feature columns
        self.feature_cols = self.data_loader.get_feature_columns(self.df)
        
        logger.info(f"✓ Data loaded: {len(self.df)} records, {len(self.feature_cols)} features")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess data"""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("="*70)
        
        # Handle missing values
        self.df = self.preprocessor.handle_missing_values(
            self.df, 
            method=config.get('preprocessing.handle_missing')
        )
        
        # Remove outliers
        if config.get('preprocessing.outlier_removal'):
            self.df = self.preprocessor.remove_outliers(
                self.df, 
                self.feature_cols,
                threshold=config.get('preprocessing.outlier_threshold')
            )
        
        logger.info("✓ Preprocessing complete")
    
    def split_data(self):
        """Split data into train/val/test sets"""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: SPLITTING DATA")
        logger.info("="*70)
        
        # Sort by date
        self.df = self.df.sort_values('date')
        
        # Calculate split dates (time series split)
        dates = self.df['date'].unique()
        n_dates = len(dates)
        
        train_end_idx = int(n_dates * config.get('data.train_split'))
        val_end_idx = int(n_dates * (config.get('data.train_split') + config.get('data.val_split')))
        
        train_end_date = dates[train_end_idx]
        val_end_date = dates[val_end_idx]
        
        # Split data
        self.train_df, self.val_df, self.test_df = self.data_loader.split_by_date(
            self.df, train_end_date, val_end_date
        )
        
        # Prepare features and labels
        self.X_train, self.y_train, self.train_metadata = self.preprocessor.prepare_features_labels(
            self.train_df, self.feature_cols
        )
        self.X_val, self.y_val, self.val_metadata = self.preprocessor.prepare_features_labels(
            self.val_df, self.feature_cols
        )
        self.X_test, self.y_test, self.test_metadata = self.preprocessor.prepare_features_labels(
            self.test_df, self.feature_cols
        )
        
        logger.info(f"✓ Data split complete")
        logger.info(f"  Train: {len(self.X_train)} samples")
        logger.info(f"  Val: {len(self.X_val)} samples")
        logger.info(f"  Test: {len(self.X_test)} samples")
    
    def feature_selection(self):
        """Select best features"""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: FEATURE SELECTION")
        logger.info("="*70)
        
        if config.get('feature_engineering.select_features'):
            # Select features
            selected_features = self.feature_selector.select_features(
                self.X_train.values, 
                self.y_train.values, 
                self.feature_cols
            )
            
            # Transform datasets
            feature_indices = [self.feature_cols.index(f) for f in selected_features if f in self.feature_cols]
            
            self.X_train = self.X_train.iloc[:, feature_indices]
            self.X_val = self.X_val.iloc[:, feature_indices]
            self.X_test = self.X_test.iloc[:, feature_indices]
            
            self.feature_cols = selected_features
            
            # Save feature selector
            self.feature_selector.save('../models/feature_selector.pkl')
        else:
            logger.info("Feature selection disabled in config")
    
    def scale_features(self):
        """Scale features"""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: SCALING FEATURES")
        logger.info("="*70)
        
        scaling_method = config.get('preprocessing.scaling_method')
        
        scaled_data = self.preprocessor.scale_features(
            self.X_train.values,
            self.X_val.values,
            self.X_test.values,
            method=scaling_method
        )
        
        # Convert back to DataFrames
        self.X_train_scaled = pd.DataFrame(scaled_data['train'], columns=self.feature_cols)
        self.X_val_scaled = pd.DataFrame(scaled_data['val'], columns=self.feature_cols)
        self.X_test_scaled = pd.DataFrame(scaled_data['test'], columns=self.feature_cols)
        
        # Save preprocessor
        self.preprocessor.feature_cols = self.feature_cols
        self.preprocessor.save_preprocessor('../models/preprocessor.pkl')
        
        logger.info("✓ Feature scaling complete")
    
    def train_models(self):
        """Train all enabled models"""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: TRAINING MODELS")
        logger.info("="*70)
        
        # Random Forest
        if config.get('models.random_forest.enabled'):
            logger.info("\nTraining Random Forest...")
            self.models['random_forest'] = RandomForestModel()
            self.models['random_forest'].train(
                self.X_train_scaled.values, 
                self.y_train.values,
                self.X_val_scaled.values,
                self.y_val.values
            )
            self.models['random_forest'].save('../models/saved_models/random_forest.pkl')
        
        # XGBoost
        if config.get('models.xgboost.enabled'):
            logger.info("\nTraining XGBoost...")
            self.models['xgboost'] = XGBoostModel()
            self.models['xgboost'].train(
                self.X_train_scaled.values, 
                self.y_train.values,
                self.X_val_scaled.values,
                self.y_val.values
            )
            self.models['xgboost'].save('../models/saved_models/xgboost.pkl')
        
        # LightGBM
        if config.get('models.lightgbm.enabled'):
            logger.info("\nTraining LightGBM...")
            self.models['lightgbm'] = LightGBMModel()
            self.models['lightgbm'].train(
                self.X_train_scaled.values, 
                self.y_train.values,
                self.X_val_scaled.values,
                self.y_val.values
            )
            self.models['lightgbm'].save('../models/saved_models/lightgbm.pkl')
        
        # Ensemble
        if config.get('models.ensemble.enabled') and len(self.models) >= 2:
            logger.info("\nTraining Ensemble...")
            self.models['ensemble'] = EnsembleModel()
            self.models['ensemble'].train(
                self.X_train_scaled.values, 
                self.y_train.values,
                self.X_val_scaled.values,
                self.y_val.values
            )
            self.models['ensemble'].save('../models/saved_models/ensemble.pkl')
        
        logger.info(f"\n✓ Trained {len(self.models)} models")
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        logger.info("\n" + "="*70)
        logger.info("STEP 7: EVALUATING MODELS")
        logger.info("="*70)
        
        for model_name, model in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING: {model_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled.values)
            y_pred_proba = model.predict_proba(self.X_test_scaled.values)
            
            # Evaluate
            results = self.evaluator.evaluate(
                self.y_test.values, 
                y_pred, 
                y_pred_proba
            )
            
            self.results[model_name] = results
            
            # Plot confusion matrix
            self.evaluator.plot_confusion_matrix(
                results['confusion_matrix'],
                save_path=f'../models/saved_models/{model_name}_confusion_matrix.png'
            )
            
            # Feature importance (if available)
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                self.evaluator.plot_feature_importance(
                    feature_importance,
                    self.feature_cols,
                    top_n=20,
                    save_path=f'../models/saved_models/{model_name}_feature_importance.png'
                )
            
            # Backtest
            actual_returns = self.test_df['future_return'].values
            backtest_results, backtest_df = self.evaluator.backtest_strategy(
                y_pred,
                actual_returns,
                self.test_metadata
            )
            
            self.results[model_name]['backtest'] = backtest_results
    
    def save_results(self):
        """Save training results"""
        logger.info("\n" + "="*70)
        logger.info("STEP 8: SAVING RESULTS")
        logger.info("="*70)
        
        # Create results summary
        summary = []
        for model_name, results in self.results.items():
            summary.append({
                'model': model_name,
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'f1_weighted': results['f1_weighted'],
                'strategy_return': results['backtest']['total_return'],
                'sharpe_ratio': results['backtest']['sharpe_ratio'],
                'win_rate': results['backtest']['win_rate']
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('../models/saved_models/results_summary.csv', index=False)
        
        logger.info("\nResults Summary:")
        print(summary_df.to_string(index=False))
        
        # Save complete results
        joblib.dump(self.results, '../models/saved_models/complete_results.pkl')
        
        logger.info("\n✓ Results saved to ../models/saved_models/")
    
    def run(self):
        """Run complete training pipeline"""
        logger.info("\n" + "="*70)
        logger.info("STOCK BULL - ML TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run pipeline
            self.load_data()
            self.preprocess_data()
            self.split_data()
            self.feature_selection()
            self.scale_features()
            self.train_models()
            self.evaluate_models()
            self.save_results()
            
            logger.info("\n" + "="*70)
            logger.info("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            logger.error(f"\n✗ Training pipeline failed: {e}")
            raise


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()