import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Select most important features for modeling
    """
    
    def __init__(self, method='importance', max_features=50):
        self.method = method
        self.max_features = max_features
        self.selected_features = None
        self.feature_scores = None
        
    def select_by_importance(self, X_train, y_train, feature_names):
        """
        Select features using Random Forest feature importance
        """
        logger.info(f"Selecting top {self.max_features} features by importance...")
        
        # Train a Random Forest to get feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top features
        self.selected_features = feature_importance_df.head(self.max_features)['feature'].tolist()
        self.feature_scores = feature_importance_df
        
        logger.info(f"✓ Selected {len(self.selected_features)} features")
        logger.info(f"Top 10 features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.selected_features
    
    def select_by_correlation(self, X_train, y_train, feature_names, threshold=0.8):
        """
        Remove highly correlated features
        """
        logger.info(f"Removing highly correlated features (threshold: {threshold})...")
        
        # Calculate correlation matrix
        X_df = pd.DataFrame(X_train, columns=feature_names)
        corr_matrix = X_df.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        self.selected_features = [f for f in feature_names if f not in to_drop]
        
        logger.info(f"✓ Removed {len(to_drop)} highly correlated features")
        logger.info(f"  Remaining features: {len(self.selected_features)}")
        
        return self.selected_features
    
    def select_by_mutual_info(self, X_train, y_train, feature_names):
        """
        Select features using mutual information
        """
        logger.info(f"Selecting top {self.max_features} features by mutual information...")
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        
        # Create DataFrame
        feature_mi_df = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        # Select top features
        self.selected_features = feature_mi_df.head(self.max_features)['feature'].tolist()
        self.feature_scores = feature_mi_df
        
        logger.info(f"✓ Selected {len(self.selected_features)} features")
        
        return self.selected_features
    
    def select_features(self, X_train, y_train, feature_names):
        """
        Main feature selection method
        """
        if self.method == 'importance':
            return self.select_by_importance(X_train, y_train, feature_names)
        elif self.method == 'correlation':
            return self.select_by_correlation(X_train, y_train, feature_names)
        elif self.method == 'mutual_info':
            return self.select_by_mutual_info(X_train, y_train, feature_names)
        else:
            logger.warning(f"Unknown method: {self.method}. Using all features.")
            return feature_names
    
    def transform(self, X, feature_names):
        """
        Transform dataset to use only selected features
        """
        if self.selected_features is None:
            logger.warning("No features selected yet. Using all features.")
            return X
        
        # Get indices of selected features
        feature_indices = [feature_names.index(f) for f in self.selected_features if f in feature_names]
        
        return X[:, feature_indices]
    
    def save(self, path='./models/feature_selector.pkl'):
        """
        Save feature selector
        """
        joblib.dump({
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'method': self.method
        }, path)
        logger.info(f"✓ Feature selector saved to {path}")
    
    def load(self, path='./models/feature_selector.pkl'):
        """
        Load feature selector
        """
        data = joblib.load(path)
        self.selected_features = data['selected_features']
        self.feature_scores = data['feature_scores']
        self.method = data['method']
        logger.info(f"✓ Feature selector loaded from {path}")


if __name__ == "__main__":
    # Test feature selection
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=100, 
                               n_informative=20, n_redundant=30, 
                               random_state=42)
    feature_names = [f'feature_{i}' for i in range(100)]
    
    # Test feature selector
    selector = FeatureSelector(method='importance', max_features=30)
    selected = selector.select_features(X, y, feature_names)
    
    print(f"\nSelected {len(selected)} features")
    print(f"Top 10: {selected[:10]}")