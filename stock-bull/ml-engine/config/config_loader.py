import yaml
import os
from pathlib import Path

class Config:
    """
    Configuration loader for ML Engine
    """
    
    def __init__(self, config_path='config/model_config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self):
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation
        Example: config.get('models.random_forest.n_estimators')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key):
        return self.config[key]


# Global config instance
config = Config()