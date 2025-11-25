"""
Training script for Cervical Cancer Risk Prediction Model.
Properly saves scaler bundle with all required metadata.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "kag_risk_factors_cervical_cancer.csv"
MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(exist_ok=True)

print("Loading cervical cancer dataset...")
df = pd.read_csv(DATASET_PATH)

# Preprocessing
print("Preprocessing data...")
df = df.drop(['STDs: Time since first diagnosis'], axis=1)
df = df.drop(['STDs: Time since last diagnosis'], axis=1)
df = df.replace('?', np.nan)
df = df.drop_duplicates()
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values
df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
df['Smokes'] = df['Smokes'].fillna(1)
df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
df['IUD'] = df['IUD'].fillna(0)
df['IUD (years)'] = df['IUD (years)'].fillna(0)
df['STDs'] = df['STDs'].fillna(1)
df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())
df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())
df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())
df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())
df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())
df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())

# Define target and features
y = df["Biopsy"]
X = df.drop(["Biopsy"], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("Training Bagging Classifier...")
base_decision_tree = DecisionTreeClassifier(max_depth=500)
bagging_model = BaggingClassifier(
    base_estimator=base_decision_tree,
    n_estimators=10,
    random_state=42
)
bagging_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = bagging_model.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = MODELS_DIR / "cc_bagging_model"
joblib.dump(bagging_model, model_path)
print(f"\nModel saved to {model_path}")

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
print(f"Scaler bundle saved to {scaler_path}")
print(f"Features: {len(feature_names)}")
print("Training complete!")

