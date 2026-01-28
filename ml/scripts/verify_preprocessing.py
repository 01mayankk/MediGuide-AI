"""
verify_preprocessing.py

Verification script for the preprocessing pipeline.

Purpose:
- Ensure preprocessing runs end-to-end without errors
- Validate output shapes and data integrity
- Catch silent bugs before model training

Design principles:
- Fast
- Deterministic
- Fail loudly if something is wrong
"""

# ============================================================
# Imports
# ============================================================

from pathlib import Path
import numpy as np

# Import preprocessing from proper package path
# This works cleanly for:
# - VS Code (Pylance)
# - CLI execution
# - Future CI pipelines
from ml.src.preprocessing import preprocess_training_data


# ============================================================
# Resolve dataset path
# ============================================================

# Project root = two levels up from this file
# ml/scripts/verify_preprocessing.py → project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to raw dataset
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "mediguide-ai.csv"


# ============================================================
# Verification logic
# ============================================================

def verify_preprocessing_pipeline() -> None:
    """
    Run a complete verification of the preprocessing pipeline.

    This function intentionally:
    - does NOT train a model
    - does NOT save artifacts
    - ONLY validates preprocessing correctness

    If any assertion fails, execution stops immediately.
    """

    print("🔍 Verifying preprocessing pipeline...")

    # --------------------------------------------------------
    # Step 1: Run preprocessing end-to-end
    # --------------------------------------------------------
    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    ) = preprocess_training_data(str(DATA_PATH))

    # --------------------------------------------------------
    # Step 2: Shape consistency checks
    # --------------------------------------------------------
    print("✔ Checking shape consistency...")

    assert X_train.shape[0] == y_train.shape[0], (
        "Mismatch between X_train and y_train row counts"
    )

    assert X_test.shape[0] == y_test.shape[0], (
        "Mismatch between X_test and y_test row counts"
    )

    assert X_train.shape[1] == X_test.shape[1], (
        "Feature count mismatch between train and test"
    )

    # --------------------------------------------------------
    # Step 3: Missing value checks
    # --------------------------------------------------------
    print("✔ Checking for missing values...")

    assert not X_train.isnull().any().any(), (
        "NaN values detected in X_train"
    )

    assert not X_test.isnull().any().any(), (
        "NaN values detected in X_test"
    )

    # --------------------------------------------------------
    # Step 4: Scaler validity checks
    # --------------------------------------------------------
    print("✔ Checking scaler object...")

    assert scaler is not None, "Scaler was not returned"

    # StandardScaler is considered fitted if mean_ exists
    assert hasattr(scaler, "mean_"), (
        "Scaler is not fitted (mean_ attribute missing)"
    )

    # --------------------------------------------------------
    # Step 5: Numerical sanity checks
    # --------------------------------------------------------
    print("✔ Checking for infinite values...")

    assert np.isfinite(X_train.values).all(), (
        "Infinite values found in X_train"
    )

    assert np.isfinite(X_test.values).all(), (
        "Infinite values found in X_test"
    )

    # --------------------------------------------------------
    # Final success message
    # --------------------------------------------------------
    print("✅ Preprocessing verification PASSED")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    verify_preprocessing_pipeline()
