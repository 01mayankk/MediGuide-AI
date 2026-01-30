"""
health.py

Health and readiness endpoints.
"""

from fastapi import APIRouter, HTTPException
from backend.app.utils.health_check import (
    is_service_healthy,
    is_service_ready,
)

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Service liveness check")
def health():
    if not is_service_healthy():
        raise HTTPException(status_code=500, detail="Service unhealthy")
    return {"status": "ok"}


@router.get("/ready", summary="Service readiness check")
def readiness():
    if not is_service_ready():
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}
