"""Common Pydantic response models and primitives."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe response."""
    status: str = Field(default="ok", description="Process liveness status")
    app: str = Field(..., description="Application name")
    environment: str = Field(..., description="Running environment")


class DependencyStatus(BaseModel):
    """Health status for an individual external dependency."""
    status: str = Field(..., description="'ok' or 'error'")
    message: str = Field(..., description="High-level status summary")


class ReadyResponse(BaseModel):
    """Readiness probe response with dependency checks."""
    status: str = Field(..., description="'ready' or 'not_ready'")
    dependencies: Dict[str, DependencyStatus] = Field(..., description="Component readiness statuses")


class ErrorResponse(BaseModel):
    """Standardized API error response format."""
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error description")
    detail: Optional[Any] = Field(default=None, description="Optional diagnostic details")


class DeletionResponse(BaseModel):
    """Response returned when an indexed document is deleted."""
    document_id: str = Field(..., description="Document identifier that was targeted")
    deleted: bool = Field(..., description="Whether records were deleted")
    points_deleted: int = Field(..., description="Number of vector points removed")
    message: str = Field(..., description="Status message")

