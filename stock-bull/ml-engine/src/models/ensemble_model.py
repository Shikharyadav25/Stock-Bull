from sklearn.ensemble import VotingClassifier
import numpy as np
import logging
import sys
sys.path.append('../..')
from src.models.base_model import BaseModel
from src.models.random_forest_model import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple classifiers
    """
    
    def __init__(self):
        super().__init__("Ensemble")
        self.base_models = {}
        self.build_model()
    
    def build_model(self):
        """Build ensemble model"""
        logger.info("Building ensemble model...")
        
        # Create base models
        self.base_models['rf'] = RandomForestModel()
        self.base_models['xgb'] = XGBoostModel()
        self.base_models['lgbm'] = LightGBMModel()
        
        # Get weights from config
        weights = config.get('models.ensemble.weights', [0.33, 0.33, 0.34])
        voting_type = config.get('models.ensemble.voting_type', 'soft')
        
        # Create voting classifier
        estimators = [
            ('rf', self.base_models['rf'].model),
            ('xgb', self.base_models['xgb'].model),
            ('lgbm', self.base_models['lgbm'].model)
        ]
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=voting_type,
            weights=weights[:3],
            n_jobs=-1
        )
        
        logger.info(f"✓ Ensemble model built with {len(estimators)} base models")
        logger.info(f"  Voting type: {voting_type}")
        logger.info(f"  Weights: {weights[:3]}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the ensemble model"""
        logger.info(f"Training {self.name}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Training accuracy
        train_acc = self.model.score(X_train, y_train)
        logger.info(f"✓ Training accuracy: {train_acc:.4f}")
        
        # Validation accuracy
        if X_val is not None and y_val is not None:
            val_acc = self.model.score(X_val, y_val)
            logger.info(f"✓ Validation accuracy: {val_acc:.4f}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_classes=5, n_informative=15, 
                               random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = EnsembleModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    print(f"\nPredictions: {predictions[:10]}")
    
    # Probabilities
    probabilities = model.predict_proba(X_test)
    print(f"\nProbabilities shape: {probabilities.shape}")
    print(f"Sample probabilities:\n{probabilities[:5]}")