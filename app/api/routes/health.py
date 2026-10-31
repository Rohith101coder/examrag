"""Health and readiness probe endpoints."""

import logging

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_vector_store
from app.api.schemas.common import DependencyStatus, HealthResponse, ReadyResponse
from app.core.config import Settings, get_settings
from app.infrastructure.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Indicates whether the application process is running. Never checks external dependencies."
)
def liveness(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        environment=settings.APP_ENV
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    description="Verifies whether external dependencies (Qdrant Cloud, OpenRouter credentials) are accessible."
)
def readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store)
) -> ReadyResponse:
    dependencies: dict[str, DependencyStatus] = {}
    is_ready = True

    # 1. Verify Qdrant Cloud connectivity
    try:
        qdrant_ok = vector_store.health_check()
        if qdrant_ok:
            dependencies["qdrant"] = DependencyStatus(status="ok", message="Connected to Qdrant Cloud cluster")
        else:
            dependencies["qdrant"] = DependencyStatus(status="error", message="Failed to reach Qdrant cluster")
            is_ready = False
    except Exception as e:
        logger.warning("Qdrant readiness probe exception: %s", str(e))
        dependencies["qdrant"] = DependencyStatus(status="error", message="Qdrant connection error")
        is_ready = False

    # 2. Verify OpenRouter configuration
    if settings.OPENROUTER_API_KEY and not settings.OPENROUTER_API_KEY.startswith("your-"):
        dependencies["openrouter"] = DependencyStatus(status="ok", message="OpenRouter credentials configured")
    else:
        dependencies["openrouter"] = DependencyStatus(status="error", message="OPENROUTER_API_KEY missing or placeholder")
        is_ready = False

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", dependencies=dependencies)

    return ReadyResponse(status="ready", dependencies=dependencies)

