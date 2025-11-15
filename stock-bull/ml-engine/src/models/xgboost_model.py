import xgboost as xgb
import numpy as np
import logging
import sys
sys.path.append('../..')
from src.models.base_model import BaseModel
from config.config_loader import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost Classifier for stock classification
    """
    
    def __init__(self):
        super().__init__("XGBoost")
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model"""
        model_config = config.get('models.xgboost')
        
        self.model = xgb.XGBClassifier(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            learning_rate=model_config['learning_rate'],
            subsample=model_config['subsample'],
            colsample_bytree=model_config['colsample_bytree'],
            objective=model_config['objective'],
            num_class=model_config['num_class'],
            random_state=model_config['random_state'],
            tree_method='hist',
            eval_metric='mlogloss'
        )
        
        logger.info(f"✓ {self.name} model built")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        logger.info(f"Training {self.name}...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
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
    model = XGBoostModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions = model.predict(X_test)
    print(f"\nPredictions: {predictions[:10]}")