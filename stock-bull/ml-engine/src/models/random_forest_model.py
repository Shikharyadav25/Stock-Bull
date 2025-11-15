from sklearn.ensemble import RandomForestClassifier
import numpy as np
import logging
import sys
sys.path.append('../..')
from src.models.base_model import BaseModel
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest Classifier for stock classification
    """
    
    def __init__(self):
        super().__init__("RandomForest")
        self.build_model()
    
    def build_model(self):
        """Build Random Forest model"""
        model_config = config.get('models.random_forest')
        
        self.model = RandomForestClassifier(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            min_samples_split=model_config['min_samples_split'],
            min_samples_leaf=model_config['min_samples_leaf'],
            max_features=model_config['max_features'],
            random_state=model_config['random_state'],
            n_jobs=model_config['n_jobs']
        )
        
        logger.info(f"✓ {self.name} model built with config:")
        for key, value in model_config.items():
            logger.info(f"  {key}: {value}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
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
    model = RandomForestModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    print(f"\nPredictions: {predictions[:10]}")
    
    # Feature importance
    importance = model.get_feature_importance()
    print(f"\nFeature importance: {importance[:10]}")