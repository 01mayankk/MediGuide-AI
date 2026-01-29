"""
verify_preprocessing.py

Verification script for the preprocessing pipeline.

Purpose:
- Ensure preprocessing runs end-to-end without errors
- Validate output shapes and data integrity
- Catch silent bugs before model training

How to run (from project root):
    python -m ml.scripts.verify_preprocessing

IMPORTANT:
- `ml` is a Python package, not just a folder
- Always use `-m` to run this module correctly

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

def verify_preprocessing_pipeline(verbose: bool = True) -> None:
    """
    Run a complete verification of the preprocessing pipeline.

    Parameters
    ----------
    verbose : bool
        If True, prints helpful summaries for developers.
        If False, runs in quiet (CI-style) mode.

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
    assert X_train.shape[0] == y_train.shape[0], (
        "Mismatch between X_train and y_train row counts"
    )

    assert X_test.shape[0] == y_test.shape[0], (
        "Mismatch between X_test and y_test row counts"
    )

    assert X_train.shape[1] == X_test.shape[1], (
        "Feature count mismatch between train and test"
    )

    if verbose:
        print(f"✔ X_train shape: {X_train.shape}")
        print(f"✔ X_test  shape: {X_test.shape}")
        print(f"✔ Number of features: {X_train.shape[1]}")

    # --------------------------------------------------------
    # Step 3: Missing value checks
    # --------------------------------------------------------
    assert not X_train.isnull().any().any(), (
        "NaN values detected in X_train"
    )

    assert not X_test.isnull().any().any(), (
        "NaN values detected in X_test"
    )

    if verbose:
        print("✔ No missing values detected")

    # --------------------------------------------------------
    # Step 4: Target distribution sanity (informative)
    # --------------------------------------------------------
    if verbose:
        print("✔ Target distribution (training data):")
        print(y_train.value_counts(normalize=True).round(3))

    # --------------------------------------------------------
    # Step 5: Scaler validity checks
    # --------------------------------------------------------
    assert scaler is not None, "Scaler was not returned"

    # StandardScaler is fitted if mean_ exists
    assert hasattr(scaler, "mean_"), (
        "Scaler is not fitted (mean_ attribute missing)"
    )

    if verbose:
        print("✔ Scaler fitted successfully")
        print("  • Mean (first 5 features):", scaler.mean_[:5])
        print("  • Std  (first 5 features):", np.sqrt(scaler.var_)[:5])

    # --------------------------------------------------------
    # Step 6: Numerical sanity checks
    # --------------------------------------------------------
    assert np.isfinite(X_train.values).all(), (
        "Infinite values found in X_train"
    )

    assert np.isfinite(X_test.values).all(), (
        "Infinite values found in X_test"
    )

    print("✅ Preprocessing verification PASSED")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    verify_preprocessing_pipeline(verbose=True)
