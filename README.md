# Machine Learning for Early Cancer Prediction

## 1. Introduction:
In contemporary healthcare, early cancer detection is of paramount importance. Identifying cancerous conditions at an early stage significantly improves patient outcomes and enables more effective treatment strategies. This project aims to leverage the potential of machine learning and deep learning methodologies to predict three prominent types of cancers: oral, cervical, and brain tumors.

## 2. Rationale:
The project is motivated by the transformative capabilities of machine learning in healthcare. Early detection empowers medical professionals to initiate interventions at the earliest stages of cancer, increasing the likelihood of successful treatment and improving overall patient prognosis. By employing advanced computational techniques, we intend to develop a sophisticated predictive model capable of analyzing medical images and risk factor data to identify potential cases of oral, cervical, and brain cancers in their early stages.

## 3. Objectives:
- Develop and fine-tune machine learning models for the early detection of oral, cervical, and brain tumors.
- Utilize comprehensive datasets, comprising both image and risk factor data, to train and validate the predictive models.
- Evaluate the performance of the models using metrics such as accuracy, sensitivity, and specificity.
- Establish a foundation for the seamless integration of predictive models into clinical practice, enhancing diagnostic capabilities in cancer detection.

## GitHub Repository Readme:

### Project Structure:
- `app/` – Streamlit UI, preprocessing helpers, and lightweight inference utilities.
- `Dataset/` – Brain MRI, oral images, and cervical risk-factor CSVs sourced from Kaggle.
- `Models/` – Serialized artifacts (`keras`, `joblib`) used by the demo.
- `Notebooks/` – Exploratory training notebooks for each disease vertical.
- `requirements.lock` – Fully pinned dependency set for reproducible setups.

## Quick Start
1. Create and activate a Python 3.11 virtual environment.
2. Install pinned dependencies:  
   `pip install -r requirements.lock`
3. Download/verify the `Dataset/` and `Models/` folders from the release bundle or Kaggle sources described in each notebook.

## Running the Streamlit Studio
```
streamlit run app/main.py
```
The app exposes three tabs:
- **Brain MRI** – CNN inference on axial slices (expects 180×180 RGB crops; the loader auto-resizes and validates uploads).
- **Cervical risk** – Bagging ensemble on 33 risk-factor features. Supports manual entry and CSV batch scoring with automatic scaling.
- **Oral screening** – Random Forest image classifier with on-the-fly resizing and probability visualization.

Invalid uploads and corrupted CSVs are surfaced directly in the UI with actionable error messages, and all model loading is cached for responsive interaction.

## Testing
Lightweight preprocessing tests are included. Run them via:
```
python -m pytest
```

## Re-training the Models

### Using Training Scripts (Recommended)
The project includes production-ready training scripts in `scripts/`:

```bash
# Train all models
python scripts/train_all.py

# Or train individually
python scripts/train_cervical.py
python scripts/train_brain.py
python scripts/train_oral.py
```

These scripts:
- Properly normalize images (divide by 255 for brain/oral models)
- Create complete scaler bundles with metadata for cervical cancer
- Save models in the correct format expected by the app
- Include proper evaluation metrics

### Using Notebooks
The original exploratory notebooks in `Notebooks/` are available for experimentation. After re-training:
1. Save the refreshed artifacts into `Models/`.
2. Ensure cervical scaler bundle includes: `scaler`, `feature_names`, `defaults`, `min`, `max`.
3. Re-run the Streamlit app to validate predictions end-to-end.

## Features

### Questionnaires
All three cancer types now include risk factor questionnaires:
- **Brain Tumor**: Demographics, symptoms (headaches, seizures, vision problems), family history, and lifestyle factors
- **Oral Cancer**: Demographics, lifestyle (smoking, alcohol, betel quid), oral health, symptoms, and medical history
- **Cervical Cancer**: Comprehensive 33-feature questionnaire covering demographics, lifestyle, STD history, and diagnostics

Questionnaires are available in separate tabs alongside image analysis for each cancer type.

### MedGemma 4B AI Interpretation
The application now includes **MedGemma 4B** integration for intelligent interpretation of model predictions and questionnaire data:

- **Automatic Interpretation**: After each prediction or questionnaire submission, MedGemma provides patient-friendly explanations of the results
- **Comprehensive Analysis**: Interprets both model outputs (probabilities, predictions) and risk factor questionnaire responses
- **Context-Aware**: Combines image analysis results with questionnaire data when both are available for more holistic interpretations
- **Fallback Support**: If MedGemma model is unavailable, the app gracefully falls back to informative text-based interpretations

**How it works:**
1. After running a prediction (image analysis) or submitting a questionnaire, an "AI-Powered Interpretation" section appears
2. MedGemma analyzes the prediction probabilities, confidence scores, and questionnaire responses
3. Provides clear explanations of what the results mean, contributing factors, and general recommendations

**Note**: The MedGemma model will be downloaded from Hugging Face on first use. For faster performance, a GPU is recommended but not required. The app works with CPU-only setups as well.

## Issues & Contributions
- Open an issue for bugs, feature ideas, or model-training questions.
- Pull requests are welcome—please describe dataset assumptions and share evaluation metrics when proposing new models.

## License
This project is licensed under the [MIT License](LICENSE).
