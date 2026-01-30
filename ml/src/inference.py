"""
inference.py

Inference module for MediGuide AI.

Purpose:
- Load trained ML artifacts (model + scaler)
- Accept structured patient feature input
- Return prediction and risk probability

IMPORTANT:
- This file must contain NO training logic
- This file is used directly by the backend API

How this file is used:
----------------------
Imported by backend service (FastAPI) to perform predictions.
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib


# ============================================================
# Paths
# ============================================================

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Artifact paths
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# ============================================================
# Feature schema (training-time features)
# ============================================================

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# ============================================================
# Load artifacts (ONCE, at import time)
# ============================================================

try:
    MODEL = joblib.load(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(
        "Model artifacts not found. "
        "Ensure train_model.py has been run successfully."
    ) from exc


# ============================================================
# Inference function
# ============================================================

def predict_risk(input_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict patient risk based on input features.

    This function enforces:
    - strict feature presence
    - strict training-time feature order
    """

    # --------------------------------------------------------
    # Step 1: Normalize input keys (case-insensitive)
    # --------------------------------------------------------
    normalized_input = {
        key.strip().lower(): value
        for key, value in input_features.items()
    }

    # --------------------------------------------------------
    # Step 2: Map input to TRAINING feature names
    # --------------------------------------------------------
    mapped_features = {}

    for feature in FEATURE_COLUMNS:
        normalized_feature = feature.lower()

        if normalized_feature not in normalized_input:
            raise ValueError(f"Missing required feature: '{feature}'")

        mapped_features[feature] = normalized_input[normalized_feature]

    # --------------------------------------------------------
    # Step 3: FORCE DataFrame column order (CRITICAL)
    # --------------------------------------------------------
    input_df = pd.DataFrame([mapped_features])

    # This line is THE MOST IMPORTANT LINE
    # It guarantees exact training-time order
    input_df = input_df.loc[:, FEATURE_COLUMNS]

    # --------------------------------------------------------
    # Step 4: Scale input
    # --------------------------------------------------------
    scaled_input = pd.DataFrame(
        SCALER.transform(input_df),
        columns=FEATURE_COLUMNS
    )


    # --------------------------------------------------------
    # Step 5: Predict
    # --------------------------------------------------------
    predicted_class = int(MODEL.predict(scaled_input)[0])
    risk_probability = float(MODEL.predict_proba(scaled_input)[0][1])

    return {
        "predicted_class": predicted_class,
        "risk_probability": round(risk_probability, 4),
    }
