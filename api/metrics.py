import pandas as pd
import joblib
import os
import json
import numpy as np
from typing import List, Dict

from sklearn.metrics import classification_report, confusion_matrix, f1_score, brier_score_loss, roc_curve, auc
from sklearn.calibration import calibration_curve
from fastapi import APIRouter

# Define the paths for the model and test data
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logreg_pipeline.pkl")
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/test_data.pkl")

# Load the saved model and test data once on startup
try:
    clf = joblib.load(MODEL_PATH)
    test_data = joblib.load(TEST_DATA_PATH)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}. Please run the training script first.")

router = APIRouter()

@router.get("/")
def get_model_metrics():
    """
    Returns the evaluation metrics for the trained classification model on the test set.
    """
    y_test_pred = clf.predict(X_test)
    y_test_probs = clf.predict_proba(X_test)

    # Classification Report
    report = classification_report(y_test, y_test_pred, output_dict=True)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()

    # Macro F1 Score
    macro_f1 = f1_score(y_test, y_test_pred, average="macro")

    # Brier Score (for model calibration)
    brier_scores = {}
    for i, cls in enumerate(clf.classes_):
        brier_scores[cls] = brier_score_loss(y_test == cls, y_test_probs[:, i])

    # Data for ROC curve (for frontend plotting)
    roc_curve_data = {}
    for i, cls in enumerate(clf.classes_):
        fpr, tpr, _ = roc_curve(y_test == cls, y_test_probs[:, i])
        roc_auc = auc(fpr, tpr)
        roc_curve_data[cls] = {
            "fpr": list(fpr),
            "tpr": list(tpr),
            "auc": float(roc_auc)
        }

    # Data for Reliability Plot (for frontend plotting)
    reliability_plot_data = {}
    for i, cls in enumerate(clf.classes_):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test == cls, y_test_probs[:, i], n_bins=10
        )
        reliability_plot_data[cls] = {
            "fraction_of_positives": list(fraction_of_positives),
            "mean_predicted_value": list(mean_predicted_value)
        }
        
    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "macro_f1_score": float(macro_f1),
        "brier_scores": {k: float(v) for k, v in brier_scores.items()},
        "evaluated_on": "test_set",
        "roc_curve_data": roc_curve_data,
        "reliability_plot_data": reliability_plot_data # Added reliability plot data
    }
