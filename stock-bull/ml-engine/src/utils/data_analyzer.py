import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    """
    Analyze and visualize training data
    """
    
    def __init__(self, df):
        self.df = df
    
    def analyze_class_balance(self):
        """
        Analyze class distribution
        """
        logger.info("\nClass Distribution Analysis:")
        logger.info("="*50)
        
        if 'label' not in self.df.columns:
            logger.warning("No 'label' column found")
            return
        
        class_counts = self.df['label'].value_counts().sort_index()
        class_pcts = self.df['label'].value_counts(normalize=True).sort_index() * 100
        
        class_names = ['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy']
        
        for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
            logger.info(f"{class_names[i]}: {count:,} ({pct:.1f}%)")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, class_counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_correlations(self, top_n=20):
        """
        Analyze feature correlations with target
        """
        logger.info("\nFeature Correlation Analysis:")
        logger.info("="*50)
        
        if 'label' not in self.df.columns:
            logger.warning("No 'label' column found")
            return
        
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['label', 'future_return']]
        
        # Calculate correlations
        correlations = self.df[numeric_cols + ['label']].corr()['label'].abs().sort_values(ascending=False)
        
        logger.info(f"\nTop {top_n} Features Correlated with Target:")
        print(correlations.head(top_n))
        
        # Plot
        plt.figure(figsize=(10, 8))
        correlations.head(top_n).plot(kind='barh')
        plt.title(f'Top {top_n} Feature Correlations with Target')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    def analyze_missing_values(self):
        """
        Analyze missing values
        """
        logger.info("\nMissing Values Analysis:")
        logger.info("="*50)
        
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        
        if len(missing) == 0:
            logger.info("No missing values found!")
            return
        
        missing_pct = (missing / len(self.df)) * 100
        
        logger.info(f"\nFeatures with missing values: {len(missing)}")
        for feature, count in missing.head(20).items():
            logger.info(f"{feature}: {count} ({missing_pct[feature]:.1f}%)")
        
        # Plot
        plt.figure(figsize=(10, 8))
        missing_pct.head(20).plot(kind='barh')
        plt.title('Top 20 Features with Missing Values')
        plt.xlabel('Percentage Missing')
        plt.tight_layout()
        plt.show()
    
    def analyze_temporal_trends(self):
        """
        Analyze temporal trends in data
        """
        if 'date' not in self.df.columns:
            logger.warning("No 'date' column found")
            return
        
        logger.info("\nTemporal Trends Analysis:")
        logger.info("="*50)
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Records over time
        records_by_date = self.df.groupby('date').size()
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Records over time
        axes[0].plot(records_by_date.index, records_by_date.values)
        axes[0].set_title('Records Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Number of Records')
        axes[0].grid(True, alpha=0.3)
        
        # Class distribution over time
        if 'label' in self.df.columns:
            class_over_time = self.df.groupby(['date', 'label']).size().unstack(fill_value=0)
            class_over_time.plot(ax=axes[1], alpha=0.7)
            axes[1].set_title('Class Distribution Over Time')
            axes[1].set_xlabel('Date')
            axes[1].set_ylabel('Count')
            axes[1].legend(title='Class', labels=['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'])
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """
        Generate comprehensive data report
        """
        logger.info("\n" + "="*70)
        logger.info("DATA ANALYSIS REPORT")
        logger.info("="*70)
        
        # Basic stats
        logger.info(f"\nDataset Shape: {self.df.shape}")
        logger.info(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        logger.info(f"Number of Stocks: {self.df['symbol'].nunique() if 'symbol' in self.df.columns else 'N/A'}")
        logger.info(f"Number of Features: {len(self.df.columns)}")
        
        # Run analyses
        self.analyze_class_balance()
        self.analyze_missing_values()
        self.analyze_feature_correlations()
        self.analyze_temporal_trends()
        
        logger.info("\n" + "="*70)


if __name__ == "__main__":
    import sys
    sys.path.append('../..')
    from src.data_preparation.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    df = loader.load_training_data()
    df = loader.calculate_future_returns(df)
    df = loader.create_labels(df)
    
    # Analyze
    analyzer = DataAnalyzer(df)
    analyzer.generate_report()