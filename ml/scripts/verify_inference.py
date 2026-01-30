"""
verify_inference.py

Verification script for the MediGuide AI inference pipeline.

Purpose:
- Ensure trained artifacts can be loaded
- Validate inference behavior with valid and invalid inputs
- Guarantee the ML-backend contract is enforced

IMPORTANT:
- This script does NOT train the model
- This script does NOT modify artifacts
- This script MUST fail loudly if assumptions are broken

How to run:
-----------
From project root:
    python -m ml.scripts.verify_inference
"""

# ============================================================
# Imports
# ============================================================

from typing import Dict, Any


# Import inference function
from ml.src.inference import predict_risk


# ============================================================
# Test Cases
# ============================================================

def valid_input_case() -> Dict[str, Any]:
    """
    Return a valid sample input that matches training schema.
    """
    return {
        "pregnancies": 2,
        "glucose": 142,
        "bloodpressure": 78,
        "skinthickness": 22,
        "insulin": 85,
        "bmi": 28.2,
        "diabetespedigreefunction": 0.52,
        "age": 45
    }


def missing_feature_case() -> Dict[str, Any]:
    """
    Input missing a required feature (should FAIL).
    """
    data = valid_input_case()
    data.pop("glucose")
    return data


def extra_feature_case() -> Dict[str, Any]:
    """
    Input containing an extra, irrelevant feature.
    This should NOT break inference.
    """
    data = valid_input_case()
    data["random_extra_feature"] = 999
    return data


# ============================================================
# Verification logic
# ============================================================

def verify_valid_inference():
    """
    Verify inference works with a correct input.
    """
    print("✔ Testing valid inference input...")

    result = predict_risk(valid_input_case())

    assert "predicted_class" in result, "Missing 'predicted_class' in output"
    assert "risk_probability" in result, "Missing 'risk_probability' in output"

    assert result["predicted_class"] in [0, 1], "Invalid predicted class"
    assert 0.0 <= result["risk_probability"] <= 1.0, "Invalid probability range"

    print("✅ Valid inference passed")


def verify_missing_feature_failure():
    """
    Verify inference fails when a required feature is missing.
    """
    print("✔ Testing missing feature failure...")

    try:
        predict_risk(missing_feature_case())
    except ValueError as exc:
        print(f"✅ Correctly failed with error: {exc}")
        return

    raise AssertionError(
        "Inference did NOT fail when required feature was missing"
    )


def verify_extra_feature_tolerance():
    """
    Verify inference ignores irrelevant extra features.
    """
    print("✔ Testing tolerance for extra features...")

    result = predict_risk(extra_feature_case())
    assert "predicted_class" in result, "Output missing predicted_class"

    print("✅ Extra feature handling passed")


# ============================================================
# Entry point
# ============================================================

def run_all_checks():
    """
    Run all inference verification tests.
    """
    print("\n🔍 Starting inference verification...\n")

    verify_valid_inference()
    verify_missing_feature_failure()
    verify_extra_feature_tolerance()

    print("\n🎉 All inference verification checks PASSED")


if __name__ == "__main__":
    run_all_checks()
