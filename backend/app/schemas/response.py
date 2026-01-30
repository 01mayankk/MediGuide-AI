"""
response.py

Pydantic schema for inference responses returned by MediGuide-AI.
"""

from pydantic import BaseModel, Field


class RiskPredictionResponse(BaseModel):
    """
    Response schema for health risk prediction.
    """

    predicted_class: int = Field(
        ...,
        description="Binary prediction output (0 = low risk, 1 = high risk)"
    )

    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability score for high-risk outcome"
    )

    risk_level: str = Field(
        ...,
        description="Human-readable risk category (Low / Medium / High)"
    )
