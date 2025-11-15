from abc import ABC, abstractmethod
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all models
    """
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self):
        """Build the model"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"{self.name} does not support probability predictions")
            return None
    
    def save(self, path):
        """Save model to disk"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"✓ {self.name} saved to {path}")
    
    def load(self, path):
        """Load model from disk"""
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"✓ {self.name} loaded from {path}")
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None