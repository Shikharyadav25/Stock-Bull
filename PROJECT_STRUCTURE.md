# Stock Bull - Project Structure Overview

## Root Level Files
- **`.gitignore`** - Git configuration to exclude files from version control (cache, venv, etc.)

---

## 📁 `data-pipeline/` 
**Purpose:** Collects stock data from multiple sources, processes it, and generates training datasets for ML models.

### Files in data-pipeline/
- **`__init__.py`** - Package initialization file
- **`.env`** - Environment variables for API keys and database credentials
- **`pipeline_scheduler.py`** - Schedules automated data collection and processing tasks
- **`QUICKSTART.md`** - Quick start guide for setting up the data pipeline
- **`README_DATA_PIPELINE.md`** - Comprehensive documentation of the data pipeline architecture
- **`requirements.txt`** - Python package dependencies (pandas, requests, sqlalchemy, etc.)
- **`run_initial_setup.py`** - One-time setup script to initialize database and collect historical data
- **`run.py`** - Main entry point for running data collection, processing, and feature generation tasks

### 📂 `collectors/`
**Purpose:** Collects raw stock data from external APIs and data sources.
- **`__init__.py`** - Package initialization
- **`stock_price_collector.py`** - Fetches historical and real-time stock prices from yfinance, NSEPy
- **`news_collector.py`** - Gathers financial news articles from NewsAPI, Google RSS feeds
- **`fundamentals_collector.py`** - Retrieves company fundamental data (P/E ratio, market cap, dividend yield, etc.)
- **`__pycache__/`** - Python compiled bytecode cache

### 📂 `config/`
**Purpose:** Configuration files and settings for the data pipeline.
- **`__init__.py`** - Package initialization
- **`config.py`** - Central configuration with database credentials, API keys, stock lists, file paths
- **`__pycache__/`** - Python compiled bytecode cache

### 📂 `logs/`
**Purpose:** Stores log files from data pipeline operations.
- **`__init__.py`** - Package initialization

### 📂 `processors/`
**Purpose:** Processes raw data and generates features for machine learning.
- **`__init__.py`** - Package initialization
- **`technical_indicators.py`** - Calculates 40+ technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA, ATR, etc.)
- **`sentiment_analyzer.py`** - Uses FinBERT AI model to analyze sentiment of news articles (scale: -1 to +1)
- **`feature_generator.py`** - Combines price, technical, sentiment, and fundamental data into training dataset
- **`__pycache__/`** - Python compiled bytecode cache

### 📂 `raw_data/`
**Purpose:** Stores raw data files before processing.
- **`__init__.py`** - Package initialization

### 📂 `processed_data/`
**Purpose:** Stores final processed datasets ready for ML training.
- **`__init__.py`** - Package initialization
- **`complete_training_dataset.csv`** - Complete processed dataset with all features (721 rows, 10 stocks, 56 features)

### 📂 `validators/`
**Purpose:** Validates data quality and completeness.
- **`__init__.py`** - Package initialization
- **`data_validator.py`** - Checks for missing values, outliers, data consistency, schema validation

### 📂 `storage/`
**Purpose:** Database management and storage operations.
- **`__init__.py`** - Package initialization
- **`database.py`** - PostgreSQL database manager (create tables, insert data, query operations)
- **`__pycache__/`** - Python compiled bytecode cache

---

## 📁 `ml-engine/`
**Purpose:** Machine learning models for stock price prediction, trained on data-pipeline outputs.

### Files in ml-engine/
- **`requirements.txt`** - ML dependencies (scikit-learn, xgboost, lightgbm, transformers, streamlit, etc.)

### 📂 `config/`
**Purpose:** ML model configuration files.
- **`__init__.py`** - Package initialization
- **`config_loader.py`** - Loads model hyperparameters from YAML configuration
- **`model_config.yaml`** - YAML file with Random Forest, XGBoost, LightGBM model hyperparameters
- **`__pycache__/`** - Python compiled bytecode cache

### 📂 `data/`
**Purpose:** Data storage for ML operations.
- **`processed/`** - Processed data for ML training
- **`raw/`** - Raw data before ML preprocessing
- **`splits/`** - Train/test/validation data splits

### 📂 `logs/`
**Purpose:** Stores ML training and prediction logs.

### 📂 `models/`
**Purpose:** Stores trained models and checkpoints.
- **`checkpoints/`** - Model checkpoints during training for resumption
- **`saved_models/`** - Final trained model files:
  - `quick_test_model.pkl` (546 KB) - Trained Random Forest model (76.38% accuracy on 10 stocks)
  - `preprocessor.pkl` (1.9 KB) - Feature scaler and column metadata

### 📂 `scripts/`
**Purpose:** Executable scripts for training and making predictions.
- **`quick_train.py`** - Fast training script on 6 months of 10 stocks data (used for quick testing)
- **`train_model.py`** - Full training script with all hyperparameter tuning
- **`predict.py`** - Simple prediction script for single stock predictions
- **`detailed_predict.py`** - Detailed prediction with sentiment analysis and technical indicators for all stocks
- **`live_predict.py`** - Real-time predictions with auto-refresh capability
- **`simple_predict.py`** - Minimal prediction example

### 📂 `src/`
**Purpose:** Core ML source code organized by functionality.

#### `data_preparation/`
- **`data_loader.py`** - Loads training dataset, calculates future returns, creates classification labels
- **`preprocessor.py`** - Handles missing values, scales features using RobustScaler, creates train/test splits
- **`__pycache__/`** - Compiled bytecode cache

#### `models/`
- **`base_model.py`** - Abstract base class for all ML models
- **`random_forest_model.py`** - Random Forest classifier (200 estimators, max_depth=15)
- **`xgboost_model.py`** - XGBoost gradient boosting model
- **`lightgbm_model.py`** - LightGBM fast gradient boosting model
- **`ensemble_model.py`** - Combines multiple models for better predictions
- **`__pycache__/`** - Compiled bytecode cache

#### `evaluation/`
- **`evaluator.py`** - Evaluates model performance (accuracy, precision, recall, F1, confusion matrix)
- **`__pycache__/`** - Compiled bytecode cache

#### `feature_engineering/`
- **`feature_selector.py`** - Selects most important features using correlation, feature importance methods

#### `prediction/`
- **`predictor.py`** - Makes predictions on new data and generates trading signals (Strong Buy/Buy/Hold/Sell/Strong Sell)
- **`__pycache__/`** - Compiled bytecode cache

#### `utils/`
- **`data_analyzer.py`** - Analyzes data distributions, correlations, statistics
- **`model_registry.py`** - Manages model versioning and persistence
- **`__pycache__/`** - Compiled bytecode cache

### 📂 `streamlit_app/`
**Purpose:** Interactive web interface for Stock Bull predictions.
- **`app.py`** - Main Streamlit application with dashboard, stock analysis, live predictions pages
- **`requirements_streamlit.txt`** - Streamlit-specific dependencies

### 📂 `notebooks/`
**Purpose:** Jupyter notebooks for exploratory data analysis and model development.
- **`01_exploratory_data_analysis.ipynb`** - EDA notebook analyzing stock data, correlations, distributions

### 📂 `tests/`
**Purpose:** Unit tests for ML models and utilities.

---

## 📁 `models/`
**Purpose:** Centralized storage for all trained models.
- **`saved_models/`** - Contains final trained models:
  - `quick_test_model.pkl` - Trained Random Forest model (76.38% accuracy)
  - `preprocessor.pkl` - Preprocessing metadata and scalers

---

## 🔑 Key Statistics
- **Total Stocks:** 10 (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, BAJFINANCE)
- **Training Data:** 721 rows covering 6 months (2025-09-11 to 2025-11-26)
- **Features:** 56 technical indicators + 7 sentiment features
- **Model Accuracy:** 76.38% on test set
- **Prediction Classes:** 5 (Strong Sell, Sell, Hold, Buy, Strong Buy)

---

## 🚀 Workflow
1. **Data Collection** → Collectors gather stock prices, news, fundamentals
2. **Data Storage** → PostgreSQL database stores raw data
3. **Data Processing** → Calculate technical indicators, sentiment analysis, features
4. **ML Training** → Train Random Forest model on 48 numeric features
5. **Predictions** → Generate buy/sell/hold signals with confidence scores
6. **Visualization** → Streamlit app displays results, charts, recommendations
