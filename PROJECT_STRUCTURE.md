# Project Structure

This document explains the organization of the Early Cancer Prediction project.

## Directory Structure

```
Early-Cancer-Prediction/
├── app/                          # Main application package
│   ├── main.py                   # Streamlit application entry point ⭐
│   ├── config.py                 # Configuration (paths, constants)
│   ├── predictors.py             # Model inference functions
│   ├── preprocessing.py          # Image/data preprocessing
│   ├── medgemma_interpreter.py   # MedGemma AI interpretation
│   ├── __init__.py
│   └── scripts/                  # Training scripts (app-specific)
│       ├── train_brain.py
│       ├── train_oral.py
│       ├── train_cervical.py
│       └── train_all.py
│
├── scripts/                      # Root-level training scripts
│   ├── train_brain.py            # Advanced brain tumor training
│   ├── train_oral.py
│   ├── train_cervical.py
│   └── train_all.py
│
├── Dataset/                      # Training and test datasets
│   ├── Brain-tumor/
│   │   ├── Training/
│   │   └── Testing/
│   ├── OralCancer/
│   └── kag_risk_factors_cervical_cancer.csv
│
├── Models/                       # Trained model files
│   ├── bt-cnn2.keras             # Brain tumor model
│   ├── oc_rf_model               # Oral cancer model
│   ├── cc_bagging_model          # Cervical cancer model
│   └── ...
│
├── Notebooks/                     # Jupyter notebooks for exploration
│   ├── Brain-Tumor-classification.ipynb
│   ├── Oral-Cancer.ipynb
│   └── Cervical-Cancer.ipynb
│
├── tests/                         # Unit tests
│   └── test_preprocessing.py
│
├── README.md                      # Main project documentation
├── requirements.txt               # Python dependencies
└── LICENSE                        # License file
```

## Key Files

### Entry Point
- **`app/main.py`** - This is the main Streamlit application. Run with:
  ```bash
  streamlit run app/main.py
  ```

### Configuration
- **`app/config.py`** - Contains all configuration constants:
  - Model paths
  - Image sizes
  - Class labels
  - App metadata

### Core Modules
- **`app/predictors.py`** - Model loading and inference functions
- **`app/preprocessing.py`** - Image preprocessing and data formatting
- **`app/medgemma_interpreter.py`** - AI-powered interpretation using MedGemma

## Training Scripts

There are training scripts in two locations:
- **`scripts/`** (root) - Recommended for training. Uses advanced architectures.
- **`app/scripts/`** - Alternative location (same functionality)

To train models:
```bash
python scripts/train_brain.py    # Train brain tumor model
python scripts/train_oral.py      # Train oral cancer model
python scripts/train_cervical.py  # Train cervical cancer model
python scripts/train_all.py        # Train all models
```

## Important Notes

1. **Only `app/main.py` should be used** - The root `main.py` was a duplicate and has been removed.

2. **The `app/` directory is a Python package** - All imports use `from app.xxx import ...`

3. **Models are saved to `Models/`** - Training scripts save models here, and the app loads from here.

4. **Dataset structure** - The app expects datasets in `Dataset/` with specific subdirectory structures.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`

