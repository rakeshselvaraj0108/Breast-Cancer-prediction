# Breast Cancer Prediction Web App

A resume-ready Flask web application that lets users upload a CSV with breast cancer features and get predictions (Benign / Malignant). The app attempts to auto-match uploaded column names to the model's expected features and provides a mapping UI when automatic matching isn't possible.

## Features
- Train a baseline model using sklearn's built-in breast cancer dataset.
- Upload CSV files for prediction.
- Automatic column name normalization & matching; manual mapping UI if needed.
- Download predictions as CSV.
- Deployable with a simple `Procfile`.

## Quick start (run locally)
1. Create a Python venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Train baseline model (creates `model_files/` artifacts):
   ```bash
   python model/train_model.py
   ```

3. Start the Flask app:
   ```bash
   python app.py
   ```

4. Open http://127.0.0.1:5000 in your browser. Upload `sample_data/sample_breast_cancer.csv` to test.

## Deploy
- On Render / Heroku or any platform that supports Python, push the repo and the app will run using the included `Procfile`.
- Make sure `model_files/` (created by step 2) is present on the server or run `python model/train_model.py` during deployment.

## Notes & Limitations
- The app uses heuristics to match uploaded column names to required features; mapping might be required for oddly-named columns.
- If your dataset uses different features than the trained model, retrain the model (see `model/train_model.py`) with your dataset for best results.

## Improvements (resume bullets)
- Added manual column mapping UI + auto-matching heuristics.
- Created robust preprocessing pipeline with scaler persistence.
- Exportable predictions and deployment-ready `Procfile`.
