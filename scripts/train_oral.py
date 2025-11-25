"""
Training script for Oral Cancer Classification Model.
Uses proper preprocessing and saves Random Forest model.
"""
import glob
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANCER_PATH = PROJECT_ROOT / "Dataset" / "OralCancer" / "cancer"
NON_CANCER_PATH = PROJECT_ROOT / "Dataset" / "OralCancer" / "non-cancer"
MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(exist_ok=True)

IMG_SIZE = (64, 64)

print("Loading oral cancer dataset...")
cancer_imgs = glob.glob(str(CANCER_PATH / "*.jpg")) + glob.glob(str(CANCER_PATH / "*.jpeg")) + glob.glob(str(CANCER_PATH / "*.png"))
non_cancer_imgs = glob.glob(str(NON_CANCER_PATH / "*.jpg")) + glob.glob(str(NON_CANCER_PATH / "*.jpeg")) + glob.glob(str(NON_CANCER_PATH / "*.png"))

print(f"Cancer images: {len(cancer_imgs)}")
print(f"Non-cancer images: {len(non_cancer_imgs)}")

dataset = []
labels = []

print("Processing images...")
for img_path in cancer_imgs:
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        dataset.append(img_array)
        labels.append(1)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

for img_path in non_cancer_imgs:
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        dataset.append(img_array)
        labels.append(0)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

dataset = np.array(dataset)
labels = np.array(labels)

print(f"Total samples: {len(dataset)}")
print(f"Shape: {dataset.shape}")

# Flatten images
X = dataset.reshape(dataset.shape[0], -1)
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-cancer", "Cancer"]))

# Save model
model_path = MODELS_DIR / "oc_rf_model"
joblib.dump(rf_model, model_path)
print(f"\nModel saved to {model_path}")
print("Training complete!")

