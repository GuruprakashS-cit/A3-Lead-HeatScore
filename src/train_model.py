# src/train_classifier.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

# 1. Load dataset
df = pd.read_csv(r"../data/leads.csv")

# Features (X) and target (y)
X = df.drop(columns=["heat_label", "lead_id"])  # drop target + ID
y = df["heat_label"]

# 2. Train/Val/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# 3. Preprocessing: One-hot encode categorical vars
categorical_features = ["source", "region", "role", "campaign", "last_touch"]
numeric_features = ["recency_days", "page_views", "prior_course_interest"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# 4. Build pipeline: preprocessing + logistic regression
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
)

# 5. Train model
clf.fit(X_train, y_train)

# 6. Validation results
y_val_pred = clf.predict(X_val)

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred))

print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_val, y_val_pred))

# 7. Macro F1 Score
macro_f1 = f1_score(y_val, y_val_pred, average="macro")
print(f"\nMacro F1 Score (Validation): {macro_f1:.3f}")

# 8. Save trained pipeline
os.makedirs("../models", exist_ok=True)
MODEL_FILE = "../models/logreg_pipeline.pkl"
joblib.dump(clf, MODEL_FILE)
print(f"\nTrained model pipeline saved to {MODEL_FILE}")
