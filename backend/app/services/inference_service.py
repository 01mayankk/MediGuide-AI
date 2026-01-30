"""
inference_service.py

Service layer responsible for executing ML inference for MediGuide-AI.

This module:
- Loads trained ML artifacts
- Delegates prediction to ML inference logic
- Acts as a clean boundary between FastAPI and ML code

IMPORTANT:
- No HTTP logic here
- No request parsing here
- No Pydantic schemas here
"""

from pathlib import Path
from typing import Dict, Any

# ML inference entrypoint
from ml.src.inference import predict_risk


# ============================================================
# Path configuration
# ============================================================

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Ensure artifacts exist early (fail fast if missing)
MODEL_ARTIFACTS_PATH = PROJECT_ROOT / "ml" / "model_artifacts"

if not MODEL_ARTIFACTS_PATH.exists():
    raise RuntimeError(
        f"ML artifacts not found at expected path: {MODEL_ARTIFACTS_PATH}"
    )


# ============================================================
# Inference Service
# ============================================================

def run_risk_prediction(input_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run ML inference using the trained MediGuide-AI model.

    Parameters
    ----------
    input_features : Dict[str, Any]
        Dictionary of validated medical input features.

    Returns
    -------
    Dict[str, Any]
        Model prediction including:
        - predicted_class
        - risk_probability
        - risk_level

    This function:
    - assumes input has already been validated
    - delegates schema enforcement to ML layer
    """

    prediction_result = predict_risk(input_features)

    # --------------------------------------------------------
    # Derive human-readable risk level
    # --------------------------------------------------------
    probability = prediction_result["risk_probability"]

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "predicted_class": prediction_result["predicted_class"],
        "risk_probability": prediction_result["risk_probability"],
        "risk_level": risk_level,
    }
