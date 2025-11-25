"""
Advanced Brain Tumor Classification Model with Transfer Learning.
Uses EfficientNetB3 as base architecture for maximum accuracy.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, applications
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "Dataset" / "Brain-tumor" / "Training"
TEST_PATH = PROJECT_ROOT / "Dataset" / "Brain-tumor" / "Testing"
MODELS_DIR = PROJECT_ROOT / "Models"
MODELS_DIR.mkdir(exist_ok=True)

# Advanced Hyperparameters
BATCH_SIZE = 16  # Reduced for larger model
IMG_HEIGHT = 224  # Increased for better detail
IMG_WIDTH = 224
EPOCHS = 50  # More epochs
INITIAL_LEARNING_RATE = 0.0001
PATIENCE = 10  # Early stopping patience

print("=" * 60)
print("Loading brain tumor dataset...")
print("=" * 60)

# Load datasets with better preprocessing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

print(f"\nClasses: {train_ds.class_names}")
print(f"Number of classes: {len(train_ds.class_names)}")

# Calculate class weights for imbalanced data
print("\nCalculating class weights...")
class_counts = {}
for images, labels in train_ds:
    for label in labels.numpy():
        class_counts[label] = class_counts.get(label, 0) + 1

total_samples = sum(class_counts.values())
class_weights = {}
for class_idx, count in class_counts.items():
    class_weights[class_idx] = total_samples / (len(class_counts) * count)
    print(f"  Class {train_ds.class_names[class_idx]}: {count} samples, weight: {class_weights[class_idx]:.4f}")

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

# Normalization
normalization_layer = layers.Rescaling(1./255)

# Build Advanced Model with Transfer Learning
print("\n" + "=" * 60)
print("Building advanced model with EfficientNetB3 transfer learning...")
print("=" * 60)

num_classes = len(train_ds.class_names)

# Option 1: Transfer Learning with EfficientNetB3 (RECOMMENDED)
def build_transfer_learning_model():
    """Build model using EfficientNetB3 as base."""
    # Load pretrained EfficientNetB3 (ImageNet weights)
    base_model = applications.EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = normalization_layer(inputs)
    x = data_augmentation(x)
    
    # Base model
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
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model

# Option 2: Deep Custom CNN (if transfer learning fails)
def build_deep_custom_model():
    """Build very deep custom CNN architecture."""
    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = normalization_layer(inputs)
    x = data_augmentation(x)
    
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
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)
    
    # Block 5
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
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, None

# Try transfer learning first, fallback to custom if needed
try:
    print("Attempting to use EfficientNetB3 transfer learning...")
    model, base_model = build_transfer_learning_model()
    use_transfer_learning = True
    print("✓ EfficientNetB3 model created successfully")
except Exception as e:
    print(f"Transfer learning failed: {e}")
    print("Falling back to deep custom CNN...")
    model, base_model = build_deep_custom_model()
    use_transfer_learning = False
    print("✓ Deep custom CNN model created")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(),
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
        filepath=str(MODELS_DIR / "bt-cnn2_best.keras"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.CSVLogger(
        filename=str(MODELS_DIR / "training_log.csv"),
        append=False
    )
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
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune with unfrozen base
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning (unfreezing base model)")
    print("=" * 60)
    
    # Unfreeze top layers of base model
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # Freeze bottom layers
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE / 10),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS - 20,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
else:
    # Single phase training for custom model
    print("\n" + "=" * 60)
    print("Training deep custom model")
    print("=" * 60)
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

# Evaluate on test set
print("\n" + "=" * 60)
print("Evaluating on test set...")
print("=" * 60)

test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(
    test_ds, verbose=1
)

print(f"\nTest Metrics:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  AUC:       {test_auc:.4f}")

# Get detailed predictions
print("\nGenerating predictions for detailed analysis...")
y_true = []
y_pred = []
y_proba = []

for images, labels in test_ds:
    y_true.extend(labels.numpy())
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_proba.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification Report
print("\n" + "=" * 60)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(
    y_true, y_pred, 
    target_names=train_ds.class_names,
    digits=4
))

# Confusion Matrix
print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
cm = confusion_matrix(y_true, y_pred)
print(cm)

# Visualize Confusion Matrix
try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_ds.class_names,
                yticklabels=train_ds.class_names)
    plt.title('Confusion Matrix - Brain Tumor Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "confusion_matrix.png"), dpi=300)
    print(f"\nConfusion matrix saved to {MODELS_DIR / 'confusion_matrix.png'}")
except Exception as e:
    print(f"Could not save confusion matrix plot: {e}")

# Per-class accuracy
print("\n" + "=" * 60)
print("PER-CLASS ACCURACY")
print("=" * 60)
for i, class_name in enumerate(train_ds.class_names):
    class_mask = y_true == i
    if np.sum(class_mask) > 0:
        class_accuracy = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
        print(f"  {class_name:15s}: {class_accuracy:.4f} ({np.sum((y_true == i) & (y_pred == i))}/{np.sum(class_mask)})")

# Save final model
model_path = MODELS_DIR / "bt-cnn2.keras"
model.save(model_path)
print(f"\n✓ Model saved to {model_path}")

# Save model info
info_path = MODELS_DIR / "bt-cnn2_info.txt"
with open(info_path, 'w') as f:
    f.write(f"Brain Tumor Classification Model - Advanced Architecture\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Architecture: {'EfficientNetB3 Transfer Learning' if use_transfer_learning else 'Deep Custom CNN'}\n")
    f.write(f"Image Size: {IMG_HEIGHT}x{IMG_WIDTH}\n")
    f.write(f"Parameters: {model.count_params():,}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Precision: {test_precision:.4f}\n")
    f.write(f"Test Recall: {test_recall:.4f}\n")
    f.write(f"Test AUC: {test_auc:.4f}\n\n")
    f.write("Per-Class Accuracy:\n")
    for i, class_name in enumerate(train_ds.class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
            f.write(f"  {class_name}: {class_accuracy:.4f}\n")

print(f"✓ Model info saved to {info_path}")
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
