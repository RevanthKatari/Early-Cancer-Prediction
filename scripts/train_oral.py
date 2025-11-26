"""
Advanced Oral Cancer Classification Model with Transfer Learning.
Uses EfficientNetB2 as base architecture for maximum accuracy.
"""
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
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

# Advanced Hyperparameters
BATCH_SIZE = 16
IMG_HEIGHT = 224  # Increased from 64 for better detail
IMG_WIDTH = 224
EPOCHS = 50
INITIAL_LEARNING_RATE = 0.0001
PATIENCE = 10

print("=" * 60)
print("Loading oral cancer dataset...")
print("=" * 60)

# Load images
cancer_imgs = glob.glob(str(CANCER_PATH / "*.jpg")) + glob.glob(str(CANCER_PATH / "*.jpeg")) + glob.glob(str(CANCER_PATH / "*.png"))
non_cancer_imgs = glob.glob(str(NON_CANCER_PATH / "*.jpg")) + glob.glob(str(NON_CANCER_PATH / "*.jpeg")) + glob.glob(str(NON_CANCER_PATH / "*.png"))

print(f"Cancer images: {len(cancer_imgs)}")
print(f"Non-cancer images: {len(non_cancer_imgs)}")

# Process images
dataset = []
labels = []

print("\nProcessing images...")
for img_path in cancer_imgs:
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32) / 255.0
        dataset.append(img_array)
        labels.append(1)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

for img_path in non_cancer_imgs:
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img, dtype=np.float32) / 255.0
        dataset.append(img_array)
        labels.append(0)
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

dataset = np.array(dataset)
labels = np.array(labels)

print(f"\nTotal samples: {len(dataset)}")
print(f"Shape: {dataset.shape}")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(f"\nClass weights: {class_weight_dict}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.25, random_state=42, stratify=labels
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Advanced Data Augmentation
print("\nSetting up data augmentation...")
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
])

# Build Advanced Model with Transfer Learning
print("\n" + "=" * 60)
print("Building advanced model with EfficientNetB2 transfer learning...")
print("=" * 60)

def build_transfer_learning_model():
    """Build model using EfficientNetB2 as base."""
    base_model = applications.EfficientNetB2(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling='avg'
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    
    # Advanced classifier head
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

def build_deep_custom_model():
    """Build deep custom CNN architecture."""
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = data_augmentation(inputs)
    
    # Block 1
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 2
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 3
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)
    
    # Block 4
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs, outputs)
    return model, None

# Try transfer learning first
try:
    print("Attempting to use EfficientNetB2 transfer learning...")
    model, base_model = build_transfer_learning_model()
    use_transfer_learning = True
    print("✓ EfficientNetB2 model created successfully")
except Exception as e:
    print(f"Transfer learning failed: {e}")
    print("Falling back to deep custom CNN...")
    model, base_model = build_deep_custom_model()
    use_transfer_learning = False
    print("✓ Deep custom CNN model created")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy', 
             keras.metrics.Precision(name='precision'),
             keras.metrics.Recall(name='recall'),
             keras.metrics.AUC(name='auc')]
)

print(f"\nModel parameters: {model.count_params():,}")
model.summary()

# Advanced Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(MODELS_DIR / "oc_rf_model_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]

# Phase 1: Train with frozen base (if transfer learning)
if use_transfer_learning and base_model is not None:
    print("\n" + "=" * 60)
    print("PHASE 1: Training with frozen base model")
    print("=" * 60)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing base model)")
    print("=" * 60)
    
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS - 20,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
else:
    print("\n" + "=" * 60)
    print("Training deep custom model")
    print("=" * 60)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )

# Evaluate
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    val_ds, verbose=1
)

print(f"\nTest Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  AUC:       {test_auc:.4f}")

# Get predictions
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=["Non-cancer", "Cancer"], digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nCONFUSION MATRIX:")
print(cm)

# Visualize Confusion Matrix
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Non-cancer", "Cancer"],
                yticklabels=["Non-cancer", "Cancer"])
    plt.title('Confusion Matrix - Oral Cancer Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "oral_confusion_matrix.png"), dpi=300)
    print(f"\nConfusion matrix saved to {MODELS_DIR / 'oral_confusion_matrix.png'}")
except Exception as e:
    print(f"Could not save confusion matrix plot: {e}")

# Save model (as .keras for consistency, but keep old name for compatibility)
model_path = MODELS_DIR / "oc_rf_model"
model.save(str(model_path) + ".keras")
# Also save as joblib for backward compatibility (convert to sklearn-like interface)
print(f"\n✓ Model saved to {model_path}.keras")

# Save model info
info_path = MODELS_DIR / "oc_rf_model_info.txt"
with open(info_path, 'w') as f:
    f.write(f"Oral Cancer Classification Model - Advanced Architecture\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Architecture: {'EfficientNetB2 Transfer Learning' if use_transfer_learning else 'Deep Custom CNN'}\n")
    f.write(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}\n")
    f.write(f"Parameters: {model.count_params():,}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test AUC: {test_auc:.4f}\n")

print(f"✓ Model info saved to {info_path}")
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
