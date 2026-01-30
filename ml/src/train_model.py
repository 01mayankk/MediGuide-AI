"""
train_model.py

Training script for the MediGuide AI model.

Purpose:
- Train the FINAL selected ML model (Random Forest)
- Use the centralized preprocessing pipeline
- Persist trained artifacts for inference and deployment
- Persist feature schema to guarantee safe inference

IMPORTANT:
- This file contains NO experimentation
- Model choice and parameters are frozen based on notebook analysis

How to run this file (RECOMMENDED):
----------------------------------
Always run this script from the PROJECT ROOT directory
using module execution so that imports resolve correctly.

1. Activate virtual environment:
   Windows (PowerShell):
       .\\venv\\Scripts\\Activate.ps1

2. Run training:
       python -m ml.src.train_model

Artifacts generated:
--------------------
ml/model_artifacts/
├── random_forest_model.pkl
├── scaler.pkl
└── feature_schema.json   <-- CRITICAL for safe inference
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json
import joblib

from sklearn.ensemble import RandomForestClassifier

# Import preprocessing pipeline
from ml.src.preprocessing import preprocess_training_data


# ============================================================
# Paths & constants
# ============================================================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mediguide-ai.csv"

# Artifact directory
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "model_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Artifact paths
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_SCHEMA_PATH = ARTIFACTS_DIR / "feature_schema.json"


# ============================================================
# Model configuration (FROZEN)
# ============================================================

def build_model() -> RandomForestClassifier:
    """
    Build the Random Forest model using frozen, approved parameters.

    Returns
    -------
    RandomForestClassifier
        Untrained Random Forest model
    """

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )


# ============================================================
# Training logic
# ============================================================

def train() -> None:
    """
    Train the Random Forest model and save artifacts.

    Steps:
    1. Run preprocessing pipeline
    2. Initialize the model
    3. Train the model
    4. Save model and scaler
    5. Save feature schema (training-time column order)
    """

    print("🚀 Starting training pipeline...")

    # --------------------------------------------------------
    # Step 1: Preprocess data
    # --------------------------------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_training_data(str(DATA_PATH))

    print(f"✔ Training samples: {X_train.shape[0]}")
    print(f"✔ Features count : {X_train.shape[1]}")

    # --------------------------------------------------------
    # Step 2: Build model
    # --------------------------------------------------------
    model = build_model()
    print("✔ Random Forest model initialized")

    # --------------------------------------------------------
    # Step 3: Train model
    # --------------------------------------------------------
    model.fit(X_train, y_train)
    print("✔ Model training complete")

    # --------------------------------------------------------
    # Step 4: Persist model artifacts
    # --------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("💾 Model saved to:", MODEL_PATH)
    print("💾 Scaler saved to:", SCALER_PATH)

    # --------------------------------------------------------
    # Step 5: Persist feature schema (CRITICAL STEP)
    # --------------------------------------------------------
    # The model + scaler expect inputs in the exact same
    # feature order they were trained on.
    #
    # Saving this schema guarantees:
    # - safe inference
    # - no silent feature misalignment
    # - hard failure if schema ever changes

    feature_schema = {
        "feature_names": list(X_train.columns)
    }

    with open(FEATURE_SCHEMA_PATH, "w") as f:
        json.dump(feature_schema, f, indent=2)

    print("💾 Feature schema saved to:", FEATURE_SCHEMA_PATH)

    print("✅ Training pipeline finished successfully")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    train()
