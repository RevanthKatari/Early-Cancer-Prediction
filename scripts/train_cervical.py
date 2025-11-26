"""
Advanced Cervical Cancer Risk Prediction Model.
Uses sophisticated ensemble methods with hyperparameter tuning.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import (
    BaggingClassifier, 
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    precision_recall_curve,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(exist_ok=True)

# Try to use enhanced dataset, fallback to original
ENHANCED_DATASET = DATASET_DIR / "cervical_cancer_enhanced.csv"
UCI_DATASET = DATASET_DIR / "cervical_cancer_uci.csv"
ORIGINAL_DATASET = DATASET_DIR / "kag_risk_factors_cervical_cancer.csv"

if ENHANCED_DATASET.exists():
    DATASET_PATH = ENHANCED_DATASET
    dataset_name = "Enhanced"
elif UCI_DATASET.exists():
    DATASET_PATH = UCI_DATASET
    dataset_name = "UCI"
else:
    DATASET_PATH = ORIGINAL_DATASET
    dataset_name = "Original Kaggle"

print("=" * 60)
print("Loading cervical cancer dataset...")
print("=" * 60)
print(f"Dataset: {dataset_name}")
print(f"Path: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"Initial dataset shape: {df.shape}")

# Advanced Preprocessing
print("\nPreprocessing data...")

# Remove time-based columns if they exist
time_cols = [col for col in df.columns if 'Time since' in col]
if time_cols:
    df = df.drop(time_cols, axis=1)
    print(f"  Dropped {len(time_cols)} time-based columns")

# Handle missing values
df = df.replace('?', np.nan)

# Convert to numeric (skip target column)
target_col = 'Biopsy' if 'Biopsy' in df.columns else df.columns[-1]
for col in df.columns:
    if col != target_col:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
if len(df) < initial_rows:
    print(f"  Removed {initial_rows - len(df)} duplicate rows")

# Advanced missing value imputation
print("Handling missing values...")
for col in df.columns:
    if df[col].isna().sum() > 0:
        if df[col].dtype in ['int64', 'float64']:
            # Use median for numeric columns
            df[col] = df[col].fillna(df[col].median())
        else:
            # Use mode for categorical
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

# Define target and features
target_col = 'Biopsy' if 'Biopsy' in df.columns else df.columns[-1]
y = df[target_col]
X = df.drop([target_col], axis=1)

# Ensure target is binary (0/1)
y = (y > 0).astype(int)

print(f"\nFeatures: {len(X.columns)}")
print(f"Samples: {len(X)}")
print(f"Positive cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
print(f"Negative cases: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")

# Data augmentation for imbalanced datasets using SMOTE
print("\n" + "=" * 60)
print("Handling Class Imbalance")
print("=" * 60)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTETomek
    
    print("Applying SMOTE for data augmentation...")
    # Use SMOTE to balance the dataset
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"  Before: {len(X)} samples ({y.sum()} positive, {(y==0).sum()} negative)")
    print(f"  After:  {len(X_resampled)} samples ({y_resampled.sum()} positive, {(y_resampled==0).sum()} negative)")
    
    X = X_resampled
    y = y_resampled
    use_smote = True
except ImportError:
    print("  SMOTE not available (install with: pip install imbalanced-learn)")
    print("  Proceeding without data augmentation...")
    use_smote = False
except Exception as e:
    print(f"  SMOTE failed: {e}")
    print("  Proceeding without data augmentation...")
    use_smote = False

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"  Train positive: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"  Test positive: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")

# Advanced Scaling (RobustScaler is better for outliers)
print("\nScaling features with RobustScaler...")
scaler = RobustScaler()  # Changed from StandardScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Advanced Ensemble Model
print("\n" + "=" * 60)
print("Building advanced ensemble model...")
print("=" * 60)

# Base estimators with optimized hyperparameters
base_dt = DecisionTreeClassifier(
    max_depth=20,  # Increased depth
    min_samples_split=5,  # More flexible
    min_samples_leaf=2,
    class_weight='balanced',  # Handle any remaining imbalance
    random_state=42
)

base_rf = RandomForestClassifier(
    n_estimators=300,  # Increased from 200
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# Bagging with more estimators
bagging_model = BaggingClassifier(
    base_estimator=base_dt,
    n_estimators=100,  # Increased from 50
    max_samples=0.8,
    max_features=0.8,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=300,  # Increased
    learning_rate=0.03,  # Lower learning rate for more stable training
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,  # Stochastic gradient boosting
    random_state=42
)

# AdaBoost
ada_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
    n_estimators=200,  # Increased
    learning_rate=0.05,  # Lower learning rate
    random_state=42
)

# Voting Ensemble (combines all models)
print("Creating voting ensemble...")
voting_model = VotingClassifier(
    estimators=[
        ('bagging', bagging_model),
        ('gradient_boosting', gb_model),
        ('adaboost', ada_model),
        ('random_forest', base_rf)
    ],
    voting='soft',  # Use probability voting
    weights=[2, 2, 1, 2]  # Weight better models more
)

# Train models
print("\nTraining ensemble models...")
print("Training Bagging Classifier...")
bagging_model.fit(X_train_scaled, y_train)

print("Training Gradient Boosting...")
gb_model.fit(X_train_scaled, y_train)

print("Training AdaBoost...")
ada_model.fit(X_train_scaled, y_train)

print("Training Random Forest...")
base_rf.fit(X_train_scaled, y_train)

print("Training Voting Ensemble...")
voting_model.fit(X_train_scaled, y_train)

# Evaluate all models
print("\n" + "=" * 60)
print("EVALUATING ALL MODELS")
print("=" * 60)

models = {
    'Bagging': bagging_model,
    'Gradient Boosting': gb_model,
    'AdaBoost': ada_model,
    'Random Forest': base_rf,
    'Voting Ensemble': voting_model
}

best_model = None
best_score = 0
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'model': model
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC:      {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    if auc > best_score:
        best_score = auc
        best_model = model
        best_name = name

print(f"\n{'='*60}")
print(f"BEST MODEL: {best_name} (AUC: {best_score:.4f})")
print(f"{'='*60}")

# Detailed evaluation of best model
print(f"\nDetailed Classification Report for {best_name}:")
print(classification_report(y_test, best_model.predict(X_test_scaled), digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
print("\nCONFUSION MATRIX:")
print(cm)

# Visualize Confusion Matrix
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f'Confusion Matrix - Cervical Cancer ({best_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "cervical_confusion_matrix.png"), dpi=300)
    print(f"\nConfusion matrix saved to {MODELS_DIR / 'cervical_confusion_matrix.png'}")
except Exception as e:
    print(f"Could not save confusion matrix plot: {e}")

# Cross-validation on best model
print(f"\nPerforming 5-fold cross-validation on {best_name}...")
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save best model
model_path = MODELS_DIR / "cc_bagging_model"
joblib.dump(best_model, model_path)
print(f"\n✓ Best model ({best_name}) saved to {model_path}")

# Create scaler bundle with metadata
feature_names = list(X.columns)
defaults = X.median().to_dict()
min_vals = X.min().to_dict()
max_vals = X.max().to_dict()

scaler_bundle = {
    "scaler": scaler,
    "feature_names": feature_names,
    "defaults": defaults,
    "min": min_vals,
    "max": max_vals,
}

scaler_path = MODELS_DIR / "cc_scaler.joblib"
joblib.dump(scaler_bundle, scaler_path)
print(f"✓ Scaler bundle saved to {scaler_path}")

# Save model info
info_path = MODELS_DIR / "cc_bagging_model_info.txt"
with open(info_path, 'w') as f:
    f.write(f"Cervical Cancer Risk Prediction Model - Advanced Ensemble\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Data Augmentation: {'SMOTE' if use_smote else 'None'}\n")
    f.write(f"Best Model: {best_name}\n")
    f.write(f"Features: {len(feature_names)}\n")
    f.write(f"Test Accuracy: {results[best_name]['accuracy']:.4f}\n")
    f.write(f"Test AUC: {results[best_name]['auc']:.4f}\n")
    f.write(f"Test F1 Score: {results[best_name]['f1']:.4f}\n")
    f.write(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
    f.write("All Models Performance:\n")
    for name, res in results.items():
        f.write(f"  {name}: AUC={res['auc']:.4f}, F1={res['f1']:.4f}\n")

print(f"✓ Model info saved to {info_path}")
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
