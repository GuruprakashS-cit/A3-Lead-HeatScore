from fastapi import APIRouter
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json

# Define the paths for the model and validation data
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logreg_pipeline.pkl")
VALIDATION_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/validation_data.pkl")

# Load the saved model and validation data once on startup
try:
    clf = joblib.load(MODEL_PATH)
    validation_data = joblib.load(VALIDATION_DATA_PATH)
    X_val = validation_data["X_val"]
    y_val = validation_data["y_val"]
except FileNotFoundError:
    raise RuntimeError("Model or validation data not found. Please run the training script first.")

router = APIRouter()

@router.get("/")
def get_model_metrics():
    """
    Returns the evaluation metrics for the trained classification model.
    """
    y_val_pred = clf.predict(X_val)

    # Get the classification report as a dictionary
    report = classification_report(y_val, y_val_pred, output_dict=True)

    # Get the confusion matrix and convert it to a list of lists for JSON serialization
    conf_matrix = confusion_matrix(y_val, y_val_pred).tolist()

    # Get the Macro F1 score
    macro_f1 = f1_score(y_val, y_val_pred, average="macro")

    return {
        "classification_report": report,
        "confusion_matrix": conf_matrix,
        "macro_f1_score": float(macro_f1),
        "evaluated_on": "validation_set"
    }