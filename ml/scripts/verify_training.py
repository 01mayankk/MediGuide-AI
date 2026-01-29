"""
verify_training.py

Verification script for trained ML artifacts.

Purpose:
- Ensure trained model and scaler can be loaded
- Verify preprocessing + inference compatibility
- Catch broken or incompatible artifacts early

Design principles:
- Fast
- Deterministic
- Fail loudly if something is wrong

How to run:
-----------
From the PROJECT ROOT directory:

    python -m ml.scripts.verify_training
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import numpy as np
import joblib

# Import preprocessing for feature consistency
from ml.src.preprocessing import preprocess_training_data


# ============================================================
# Paths
# ============================================================

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Artifact paths
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

# Dataset path (used only for verification)
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mediguide-ai.csv"


# ============================================================
# Verification logic
# ============================================================

def verify_training_artifacts() -> None:
    """
    Verify that trained artifacts are valid and usable.

    This function checks:
    1. Artifacts exist
    2. Model and scaler can be loaded
    3. Inference runs end-to-end on sample input
    """

    print("🔍 Verifying trained model artifacts...")

    # --------------------------------------------------------
    # Step 1: Check artifact existence
    # --------------------------------------------------------
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    assert SCALER_PATH.exists(), f"Scaler file not found: {SCALER_PATH}"

    print("✔ Model and scaler files found")

    # --------------------------------------------------------
    # Step 2: Load artifacts
    # --------------------------------------------------------
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("✔ Model and scaler loaded successfully")

    # --------------------------------------------------------
    # Step 3: Prepare sample input (via preprocessing)
    # --------------------------------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _
    ) = preprocess_training_data(str(DATA_PATH))

    # Take a small sample for verification
    X_sample = X_test.iloc[:5]

    print("✔ Sample input prepared")

    # --------------------------------------------------------
    # Step 4: Run inference
    # --------------------------------------------------------
    y_pred = model.predict(X_sample)
    y_prob = model.predict_proba(X_sample)

    # --------------------------------------------------------
    # Step 5: Sanity checks
    # --------------------------------------------------------
    assert y_pred.shape[0] == X_sample.shape[0], (
        "Prediction count does not match input rows"
    )

    assert y_prob.shape == (X_sample.shape[0], 2), (
        "Probability output shape is incorrect"
    )

    assert np.isfinite(y_prob).all(), (
        "Non-finite probability values detected"
    )

    print("✔ Inference sanity checks passed")

    # --------------------------------------------------------
    # Final success message
    # --------------------------------------------------------
    print("✅ Training artifact verification PASSED")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    verify_training_artifacts()
