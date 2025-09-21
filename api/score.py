from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/logreg_pipeline.pkl")
clf = joblib.load(MODEL_PATH)


# Load the saved Logistic Regression pipeline
#MODEL_PATH = "../models/logreg_pipeline.pkl"
clf = joblib.load(MODEL_PATH)


router = APIRouter()

from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

router = APIRouter()


class Lead(BaseModel):
    lead_id: int
    source: str
    recency_days: int
    region: str
    role: str
    campaign: str
    page_views: int
    last_touch: str
    prior_course_interest: int

@router.post("/")
def score_lead(lead: Lead):
    lead_df = pd.DataFrame([lead.dict()])

    # Predict probabilities
    probs = clf.predict_proba(lead_df)[0]
    classes = clf.classes_
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probs)}

    # Compute top 3 contributing features
    # 1. Get preprocessor and classifier
    preprocessor = clf.named_steps['preprocessor']
    classifier = clf.named_steps['classifier']

    # 2. Get transformed feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(preprocessor.transformers_[0][2])
    num_features = preprocessor.transformers_[1][2]
    all_features = np.concatenate([cat_features, num_features])

    # 3. Get coefficients for the predicted class
    pred_class_index = list(classes).index(prob_dict.keys().__iter__().__next__())
    coefs = classifier.coef_[pred_class_index]

    # 4. Get top 3 feature names by absolute coefficient
    top_indices = np.argsort(np.abs(coefs))[-3:][::-1]
    top_features = all_features[top_indices].tolist()

    return {
        "class": clf.predict(lead_df)[0],
        "probabilities": prob_dict,
        "top_features": top_features
    }
