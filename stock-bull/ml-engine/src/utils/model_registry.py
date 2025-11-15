import joblib
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Track and manage trained models
    """
    
    def __init__(self, registry_path='../models/model_registry.json'):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(self, model_name, model_path, metrics, metadata=None):
        """
        Register a trained model
        """
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.registry[model_id] = {
            'name': model_name,
            'path': str(model_path),
            'registered_at': datetime.now().isoformat(),
            'metrics': metrics,
            'metadata': metadata or {}
        }
        
        self._save_registry()
        logger.info(f"✓ Model registered: {model_id}")
        
        return model_id
    
    def get_best_model(self, metric='accuracy'):
        """
        Get best model based on metric
        """
        if not self.registry:
            return None
        
        best_model = max(
            self.registry.items(),
            key=lambda x: x[1]['metrics'].get(metric, 0)
        )
        
        return best_model
    
    def list_models(self):
        """
        List all registered models
        """
        if not self.registry:
            logger.info("No models registered")
            return
        
        logger.info("\nRegistered Models:")
        logger.info("="*70)
        
        for model_id, info in self.registry.items():
            logger.info(f"\n{model_id}:")
            logger.info(f"  Name: {info['name']}")
            logger.info(f"  Registered: {info['registered_at']}")
            logger.info(f"  Accuracy: {info['metrics'].get('accuracy', 'N/A'):.4f}")
            logger.info(f"  F1 Score: {info['metrics'].get('f1_macro', 'N/A'):.4f}")


if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Example: Register a model
    metrics = {
        'accuracy': 0.75,
        'f1_macro': 0.72,
        'precision_macro': 0.73
    }
    
    model_id = registry.register_model(
        'random_forest',
        '../models/saved_models/random_forest.pkl',
        metrics,
        metadata={'n_estimators': 200, 'max_depth': 15}
    )
    
    # List models
    registry.list_models()
    
    # Get best model
    best = registry.get_best_model('accuracy')
    if best:
        print(f"\nBest model: {best[0]}")