import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, class_names=None):
        if class_names is None:
            self.class_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        else:
            self.class_names = class_names
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Comprehensive evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics (better for imbalanced classes)
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            results[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0
            results[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0
            results[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0
        
        # ROC AUC (if probabilities available)
        if y_pred_proba is not None:
            try:
                # Multiclass ROC AUC (One-vs-Rest)
                results['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', average='macro')
                results['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovo', average='macro')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        
        # Log results
        self._log_results(results)
        
        return results
    
    def _log_results(self, results):
        """Log evaluation results"""
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision (macro): {results['precision_macro']:.4f}")
        logger.info(f"Recall (macro): {results['recall_macro']:.4f}")
        logger.info(f"F1 Score (macro): {results['f1_macro']:.4f}")
        
        if 'roc_auc_ovr' in results:
            logger.info(f"ROC AUC (OvR): {results['roc_auc_ovr']:.4f}")
        
        logger.info("\nPer-Class Metrics:")
        for class_name in self.class_names:
            prec = results.get(f'precision_{class_name}', 0)
            rec = results.get(f'recall_{class_name}', 0)
            f1 = results.get(f'f1_{class_name}', 0)
            logger.info(f"  {class_name}: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
        
        logger.info("\n" + results['classification_report'])
        logger.info("="*60 + "\n")
    
    def plot_confusion_matrix(self, confusion_mat, save_path=None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_importance, feature_names, top_n=20, save_path=None):
        """
        Plot feature importance
        """
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Feature importance plot saved to {save_path}")
        
        plt.show()
        
        return importance_df
    
    def plot_class_distribution(self, y_true, y_pred, save_path=None):
        """
        Plot actual vs predicted class distribution
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual distribution
        unique, counts = np.unique(y_true, return_counts=True)
        ax1.bar([self.class_names[i] for i in unique], counts, color='steelblue')
        ax1.set_title('Actual Class Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        ax2.bar([self.class_names[i] for i in unique_pred], counts_pred, color='coral')
        ax2.set_title('Predicted Class Distribution')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Class distribution plot saved to {save_path}")
        
        plt.show()
    
    def backtest_strategy(self, predictions, actual_returns, metadata):
        """
        Backtest trading strategy based on predictions
        """
        logger.info("Running backtest...")
        
        df = pd.DataFrame({
            'symbol': metadata['symbol'],
            'date': metadata['date'],
            'prediction': predictions,
            'actual_return': actual_returns,
            'close': metadata['close']
        })
        
        # Trading strategy: Buy stocks predicted as Buy/Strong Buy (classes 3, 4)
        df['signal'] = df['prediction'].apply(lambda x: 1 if x >= 3 else 0)
        
        # Calculate strategy returns
        df['strategy_return'] = df['signal'] * df['actual_return']
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['actual_return'] / 100).cumprod()
        df['strategy_cumulative'] = (1 + df['strategy_return'] / 100).cumprod()
        
        # Performance metrics
        total_return = (df['strategy_cumulative'].iloc[-1] - 1) * 100
        market_return = (df['cumulative_return'].iloc[-1] - 1) * 100
        
        # Sharpe ratio (simplified)
        returns = df['strategy_return'].dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Win rate
        winning_trades = (df[df['signal'] == 1]['actual_return'] > 0).sum()
        total_trades = (df['signal'] == 1).sum()
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'total_return': total_return,
            'market_return': market_return,
            'excess_return': total_return - market_return,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        logger.info("\nBacktest Results:")
        logger.info(f"  Strategy Return: {total_return:.2f}%")
        logger.info(f"  Market Return: {market_return:.2f}%")
        logger.info(f"  Excess Return: {results['excess_return']:.2f}%")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total Trades: {total_trades}")
        
        return results, df


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_classes=5, n_informative=15, 
                               random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_test, y_pred, y_pred_proba)
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(results['confusion_matrix'])