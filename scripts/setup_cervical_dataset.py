"""
Setup script to download and prepare the best cervical cancer dataset.
Run this before training: python scripts/setup_cervical_dataset.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
DATASET_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Cervical Cancer Dataset Setup")
print("=" * 60)

# Option 1: Try UCI ML Repository dataset
print("\n[1/3] Attempting to download UCI Cervical Cancer dataset...")
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
uci_path = DATASET_DIR / "cervical_cancer_uci.csv"

try:
    urllib.request.urlretrieve(UCI_URL, uci_path)
    df_uci = pd.read_csv(uci_path)
    print(f"✓ Downloaded UCI dataset: {df_uci.shape[0]} samples, {df_uci.shape[1]} features")
    uci_available = True
except Exception as e:
    print(f"✗ Could not download UCI dataset: {e}")
    uci_available = False

# Option 2: Enhance current dataset
print("\n[2/3] Enhancing current dataset...")
current_path = DATASET_DIR / "kag_risk_factors_cervical_cancer.csv"

if current_path.exists():
    df = pd.read_csv(current_path)
    print(f"✓ Found current dataset: {df.shape[0]} samples")
    
    # Enhanced preprocessing
    df = df.replace('?', np.nan)
    
    # Remove time columns
    time_cols = [col for col in df.columns if 'Time since' in col]
    if time_cols:
        df = df.drop(time_cols, axis=1)
    
    # Convert to numeric
    for col in df.columns:
        if col != 'Biopsy':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Advanced imputation
    for col in df.columns:
        if col != 'Biopsy' and df[col].isna().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 0)
    
    # Feature engineering
    if 'Age' in df.columns and 'Number of sexual partners' in df.columns:
        df['Age_Partners_Interaction'] = df['Age'] * df['Number of sexual partners']
    
    if 'Smokes (years)' in df.columns and 'Smokes (packs/year)' in df.columns:
        df['Smoking_Total_Exposure'] = df['Smokes (years)'] * df['Smokes (packs/year)']
    
    std_features = [col for col in df.columns if 'STDs:' in col and 'Number' not in col]
    if std_features:
        df['STD_Risk_Score'] = df[std_features].sum(axis=1)
    
    enhanced_path = DATASET_DIR / "cervical_cancer_enhanced.csv"
    df.to_csv(enhanced_path, index=False)
    print(f"✓ Enhanced dataset saved: {df.shape[0]} samples, {df.shape[1]} features")
    enhanced_available = True
else:
    print("✗ Current dataset not found")
    enhanced_available = False

# Option 3: Create synthetic dataset if both fail
print("\n[3/3] Dataset availability check...")
if uci_available:
    print("✓ UCI dataset available - will be used for training")
elif enhanced_available:
    print("✓ Enhanced dataset available - will be used for training")
else:
    print("✗ No suitable dataset found. Please download manually:")
    print("  1. UCI: https://archive.ics.uci.edu/ml/datasets/Cervical+Cancer+Risk+Factors")
    print("  2. Or place kag_risk_factors_cervical_cancer.csv in Dataset/ folder")

print("\n" + "=" * 60)
print("Setup Complete!")
print("=" * 60)
print("\nNext step: Run training with:")
print("  python scripts/train_cervical.py")

