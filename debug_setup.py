import sys
import os

print("="*70)
print("SETUP DIAGNOSTIC")
print("="*70)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check if in virtual environment
print(f"2. Virtual Environment: {'Yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'No'}")

# Check critical packages
packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm', 'sqlalchemy']
print("\n3. Installed Packages:")
for pkg in packages:
    try:
        __import__(pkg)
        print(f"   ✓ {pkg}")
    except ImportError:
        print(f"   ✗ {pkg} - NOT INSTALLED")

# Check database
print("\n4. Database Connection:")
try:
    sys.path.insert(0, 'stock-bull/data-pipeline')
    from storage.database import DatabaseManager
    db = DatabaseManager()
    print("   ✓ Database connection successful")
except Exception as e:
    print(f"   ✗ Database error: {e}")

# Check data files
print("\n5. Data Files:")
data_file = "stock-bull/data-pipeline/processed_data/complete_training_dataset.csv"
if os.path.exists(data_file):
    import pandas as pd
    df = pd.read_csv(data_file)
    print(f"   ✓ Training data exists: {len(df)} records")
else:
    print(f"   ✗ No training data found at {data_file}")

print("\n" + "="*70)
