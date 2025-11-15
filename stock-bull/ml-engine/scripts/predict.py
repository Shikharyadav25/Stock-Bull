#!/usr/bin/env python3
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def predict_latest_stocks(model_name='quick_test_model'):
    """
    Make predictions on latest stock data
    """
    try:
        from src.data_preparation.data_loader import DataLoader
        from src.prediction.predictor import StockPredictor
        
        logger.info("\n" + "="*70)
        logger.info("STOCK BULL - PREDICTION PIPELINE")
        logger.info("="*70)
        
        # Load data
        logger.info("\nLoading latest stock data...")
        loader = DataLoader()
        df = loader.load_training_data()
        
        # Get only the latest date for each stock
        latest_df = df.sort_values('date').groupby('symbol').tail(1).reset_index(drop=True)
        
        logger.info(f"Loaded latest data for {len(latest_df)} stocks")
        logger.info(f"Latest date: {latest_df['date'].max()}")
        
        # Initialize predictor
        model_path = f'../models/saved_models/{model_name}.pkl'
        
        if not os.path.exists(model_path):
            logger.error(f"\n✗ Model not found: {model_path}")
            logger.error(f"Available models:")
            models_dir = '../models/saved_models'
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    if f.endswith('.pkl'):
                        logger.error(f"  - {f}")
            else:
                logger.error(f"  Models directory not found. Please train a model first.")
            sys.exit(1)
        
        predictor = StockPredictor(model_path)
        
        # Make predictions
        logger.info("\nMaking predictions...")
        predictions = predictor.predict(latest_df)
        
        # Sort by confidence
        predictions_sorted = predictions.sort_values('confidence', ascending=False)
        
        # Display results
        logger.info("\n" + "="*70)
        logger.info("PREDICTION RESULTS")
        logger.info("="*70)
        
        for idx, row in predictions_sorted.iterrows():
            logger.info(f"\n{row['symbol']}:")
            logger.info(f"  Prediction: {row['prediction_label']}")
            logger.info(f"  Confidence: {row['confidence']:.1%}")
            
            # Show probabilities
            logger.info(f"  Probabilities:")
            for class_name in ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']:
                col_name = f'prob_{class_name}'
                if col_name in row:
                    logger.info(f"    {class_name}: {row[col_name]:.1%}")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        
        buy_stocks = predictions_sorted[predictions_sorted['prediction_label'].isin(['Buy', 'Strong Buy'])]
        hold_stocks = predictions_sorted[predictions_sorted['prediction_label'] == 'Hold']
        sell_stocks = predictions_sorted[predictions_sorted['prediction_label'].isin(['Sell', 'Strong Sell'])]
        
        if len(buy_stocks) > 0:
            logger.info(f"\n✅ STOCKS TO BUY ({len(buy_stocks)}):")
            for _, row in buy_stocks.iterrows():
                logger.info(f"  {row['symbol']}: {row['prediction_label']} ({row['confidence']:.1%})")
        
        if len(hold_stocks) > 0:
            logger.info(f"\n⏸️  STOCKS TO HOLD ({len(hold_stocks)}):")
            for _, row in hold_stocks.head(3).iterrows():
                logger.info(f"  {row['symbol']}: {row['prediction_label']} ({row['confidence']:.1%})")
        
        if len(sell_stocks) > 0:
            logger.info(f"\n❌ STOCKS TO AVOID ({len(sell_stocks)}):")
            for _, row in sell_stocks.iterrows():
                logger.info(f"  {row['symbol']}: {row['prediction_label']} ({row['confidence']:.1%})")
        
        # Save results
        os.makedirs('../data/processed', exist_ok=True)
        from datetime import datetime
        output_file = f'../data/processed/predictions_{datetime.now().strftime("%Y%m%d")}.csv'
        predictions_sorted.to_csv(output_file, index=False)
        logger.info(f"\n✓ Predictions saved to {output_file}")
        
        logger.info("\n" + "="*70)
        logger.info("🎉 PREDICTION COMPLETE!")
        logger.info("="*70)
        
        return predictions_sorted
        
    except Exception as e:
        logger.error(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make stock predictions')
    parser.add_argument('--model', type=str, default='quick_test_model', 
                       help='Model to use')
    
    args = parser.parse_args()
    
    predictions = predict_latest_stocks(model_name=args.model)