"""
predict.py

FastAPI route for MediGuide-AI risk prediction.

Responsibilities:
- Accept HTTP requests
- Validate input using Pydantic schemas
- Delegate inference to service layer
- Return structured response

IMPORTANT:
- No ML logic here
- No preprocessing here
- No business logic here
"""

from fastapi import APIRouter, HTTPException

from backend.app.schemas.request import RiskPredictionRequest
from backend.app.schemas.response import RiskPredictionResponse
from backend.app.services.inference_service import run_risk_prediction


# ============================================================
# Router setup
# ============================================================

router = APIRouter(
    prefix="/predict",
    tags=["Risk Prediction"]
)


# ============================================================
# Prediction endpoint
# ============================================================

@router.post(
    "",
    response_model=RiskPredictionResponse,
    summary="Predict diabetes risk from medical inputs"
)
def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Predict diabetes risk based on patient health metrics.

    Workflow:
    1. Request body validated automatically by Pydantic
    2. Input forwarded to inference service
    3. ML prediction returned as response model
    """

    try:
        prediction = run_risk_prediction(request.model_dump())

        return RiskPredictionResponse(**prediction)

    except Exception as exc:
        # Catch any unexpected inference errors
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )
