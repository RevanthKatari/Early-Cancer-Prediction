"""
Download and prepare a better cervical cancer dataset.
Downloads UCI Cervical Cancer Risk Factors dataset and enhances it.
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

# UCI Cervical Cancer Dataset URLs
UCI_DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"

print("=" * 60)
print("Downloading Better Cervical Cancer Dataset")
print("=" * 60)

# Try to download UCI dataset
print("\nAttempting to download UCI Cervical Cancer dataset...")
try:
    output_path = DATASET_DIR / "cervical_cancer_uci.csv"
    urllib.request.urlretrieve(UCI_DATASET_URL, output_path)
    print(f"✓ Downloaded UCI dataset to {output_path}")
    
    # Load and check
    df_uci = pd.read_csv(output_path)
    print(f"  Shape: {df_uci.shape}")
    print(f"  Columns: {len(df_uci.columns)}")
    
    # Use UCI dataset
    df = df_uci.copy()
    dataset_source = "UCI"
    
except Exception as e:
    print(f"Could not download UCI dataset: {e}")
    print("Falling back to current dataset with enhancements...")
    
    # Use current dataset
    current_path = DATASET_DIR / "kag_risk_factors_cervical_cancer.csv"
    if current_path.exists():
        df = pd.read_csv(current_path)
        dataset_source = "Kaggle (enhanced)"
    else:
        print("ERROR: No dataset found!")
        exit(1)

print(f"\nUsing dataset: {dataset_source}")
print(f"Initial shape: {df.shape}")

# Enhanced preprocessing
print("\n" + "=" * 60)
print("Enhanced Data Preprocessing")
print("=" * 60)

# Remove problematic columns
columns_to_drop = []
for col in df.columns:
    if 'Time since' in col or 'Time since first' in col or 'Time since last' in col:
        columns_to_drop.append(col)

if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)
    print(f"Dropped {len(columns_to_drop)} time-based columns")

# Handle missing values
print("\nHandling missing values...")
df = df.replace('?', np.nan)

# Convert to numeric
for col in df.columns:
    if col != 'Biopsy':  # Don't convert target yet
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows")

# Advanced imputation strategy
print("\nAdvanced missing value imputation...")
missing_before = df.isna().sum().sum()

for col in df.columns:
    if col == 'Biopsy':
        continue
    
    missing_count = df[col].isna().sum()
    if missing_count > 0:
        if df[col].dtype in ['int64', 'float64']:
            # Use median for numeric (more robust than mean)
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
        else:
            # Use mode for categorical
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna(0)

missing_after = df.isna().sum().sum()
print(f"  Filled {missing_before - missing_after} missing values")

# Data quality checks
print("\nData Quality Checks:")
print(f"  Total samples: {len(df)}")
print(f"  Total features: {len(df.columns) - 1}")  # Excluding target

if 'Biopsy' in df.columns:
    target = df['Biopsy']
    print(f"  Positive cases: {target.sum()} ({target.sum()/len(target)*100:.2f}%)")
    print(f"  Negative cases: {(target==0).sum()} ({(target==0).sum()/len(target)*100:.2f}%)")
    
    # Check for class imbalance
    if target.sum() < len(target) * 0.1:
        print("  ⚠ Warning: Severe class imbalance detected!")
    elif target.sum() < len(target) * 0.3:
        print("  ⚠ Warning: Moderate class imbalance detected")

# Feature engineering for better performance
print("\n" + "=" * 60)
print("Feature Engineering")
print("=" * 60)

# Create interaction features
if 'Age' in df.columns and 'Number of sexual partners' in df.columns:
    df['Age_Partners_Interaction'] = df['Age'] * df['Number of sexual partners']
    print("  Created: Age × Partners interaction")

if 'Smokes (years)' in df.columns and 'Smokes (packs/year)' in df.columns:
    df['Smoking_Total_Exposure'] = df['Smokes (years)'] * df['Smokes (packs/year)']
    print("  Created: Total smoking exposure")

# Create risk score features
std_features = [col for col in df.columns if 'STDs:' in col and col != 'STDs: Number of diagnosis']
if std_features:
    df['STD_Risk_Score'] = df[std_features].sum(axis=1)
    print(f"  Created: STD Risk Score (from {len(std_features)} STD features)")

# Save enhanced dataset
output_path = DATASET_DIR / "cervical_cancer_enhanced.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Enhanced dataset saved to {output_path}")
print(f"  Final shape: {df.shape}")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Samples: {len(df)}")

print("\n" + "=" * 60)
print("Dataset Preparation Complete!")
print("=" * 60)
print(f"\nEnhanced dataset ready at: {output_path}")
print("You can now train with: python scripts/train_cervical.py")

