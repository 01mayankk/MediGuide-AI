"""
request.py

Pydantic schema for validating inference requests to MediGuide-AI.

This schema is the first line of defense for the ML system.
Only clean, well-formed medical data should pass beyond this layer.
"""

from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    """
    Request schema for health risk prediction.

    Each field is required and validated to prevent
    invalid medical inputs from reaching the model.
    """

    pregnancies: int = Field(
        ...,
        ge=0,
        description="Number of times the patient has been pregnant"
    )

    glucose: float = Field(
        ...,
        gt=0,
        description="Plasma glucose concentration"
    )

    bloodpressure: float = Field(
        ...,
        gt=0,
        description="Diastolic blood pressure (mm Hg)"
    )

    skinthickness: float = Field(
        ...,
        gt=0,
        description="Triceps skin fold thickness (mm)"
    )

    insulin: float = Field(
        ...,
        ge=0,
        description="2-hour serum insulin (mu U/ml)"
    )

    bmi: float = Field(
        ...,
        gt=0,
        description="Body mass index (kg/m²)"
    )

    diabetespedigreefunction: float = Field(
        ...,
        ge=0,
        description="Diabetes pedigree function"
    )

    age: int = Field(
        ...,
        gt=0,
        description="Age of the patient in years"
    )
