"""
health_check.py

Utility functions for service health and readiness checks.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "ml" / "model_artifacts"


def is_service_healthy() -> bool:
    """
    Basic liveness check.
    Returns True if the service is running.
    """
    return True


def is_service_ready() -> bool:
    """
    Readiness check.

    Ensures:
    - ML artifacts exist
    - Model can be served
    """
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())
