# model/train_model.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model_files')
os.makedirs(MODEL_DIR, exist_ok=True)

def train_default_model():
    # Load sklearn built-in breast cancer dataset
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print("Test accuracy:", acc)
    print(classification_report(y_test, preds, target_names=data.target_names))

    # Save artifacts
    with open(os.path.join(MODEL_DIR, 'cancer_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    # Save expected feature names for downstream column matching
    feature_names = list(X.columns)
    with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    print("Saved model, scaler, and feature list to:", MODEL_DIR)

if __name__ == "__main__":
    train_default_model()
