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
Updated notebooks in `Notebooks/` walk through the training pipelines (data augmentation, hyper-parameter search, and export steps). After re-training:
1. Save the refreshed artifacts into `Models/`.
2. Re-run the Streamlit app to validate predictions end-to-end.

## Issues & Contributions
- Open an issue for bugs, feature ideas, or model-training questions.
- Pull requests are welcome—please describe dataset assumptions and share evaluation metrics when proposing new models.

## License
This project is licensed under the [MIT License](LICENSE).
