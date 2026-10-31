"""Main FastAPI application entry point with lifespan, CORS, middleware, and routes."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import get_rag_service
from app.api.routes import chat, documents, health
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.api.schemas.common import ErrorResponse
from app.core.config import get_settings
from app.core.exceptions import (
    AuthenticationError,
    CollectionDimensionMismatchError,
    ConfigurationError,
    DocumentNotFoundError,
    DocumentParsingError,
    ExamRAGException,
    InvalidPDFError,
    OversizedFileError,
    ProviderError,
    ProviderTimeoutError,
    StructuredOutputParsingError,
    VectorStoreError,
)
from app.core.logging import setup_logging
from app.services.rag_service import RAGService

settings = get_settings()
logger = setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context for startup validation and graceful shutdown."""
    logger.info("Initializing %s in [%s] mode...", settings.APP_NAME, settings.APP_ENV)

    # Validate production prerequisites
    try:
        settings.validate_production_requirements()
    except ConfigurationError as e:
        logger.critical("Startup configuration error: %s", e.message)
        raise

    logger.info("Startup complete. API V1 prefix: %s", settings.API_V1_PREFIX)
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


def create_application() -> FastAPI:
    """Factory function for FastAPI application instance."""
    app = FastAPI(
        title="ExamRAG API",
        description=(
            "Production-Minded Document-Grounded Question Answering API "
            "for B.Tech Computer Science Students using Qdrant Cloud and OpenRouter."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        lifespan=lifespan
    )

    # CORS Middleware
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Register Routers
    api_prefix = settings.API_V1_PREFIX
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(documents.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)

    # Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                error="ValidationError",
                message="Request payload failed schema validation.",
                detail=exc.errors()
            ).model_dump()
        )

    @app.exception_handler(AuthenticationError)
    async def auth_exception_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=ErrorResponse(
                error="AuthenticationError",
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(InvalidPDFError)
    @app.exception_handler(DocumentParsingError)
    async def bad_request_exception_handler(request: Request, exc: ExamRAGException) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(OversizedFileError)
    async def oversized_file_exception_handler(request: Request, exc: OversizedFileError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                error="OversizedFileError",
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(DocumentNotFoundError)
    async def not_found_exception_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                error="DocumentNotFoundError",
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(ProviderTimeoutError)
    async def provider_timeout_handler(request: Request, exc: ProviderTimeoutError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content=ErrorResponse(
                error="ProviderTimeoutError",
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(StructuredOutputParsingError)
    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ExamRAGException) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.message
            ).model_dump()
        )

    @app.exception_handler(CollectionDimensionMismatchError)
    @app.exception_handler(VectorStoreError)
    async def vectorstore_error_handler(request: Request, exc: ExamRAGException) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=type(exc).__name__,
                message="A vector store operation error occurred."
            ).model_dump()
        )

    @app.get("/", tags=["Root"])
    def root() -> dict[str, str]:
        return {
            "app": settings.APP_NAME,
            "version": "1.0.0",
            "docs": "/docs",
            "api_v1": settings.API_V1_PREFIX
        }

    @app.get(
        "/ask",
        response_model=ChatResponse,
        tags=["Chat"],
        summary="Ask question with vector, graph, or hybrid retrieval",
        description="Public endpoint supporting mode=vector|graph|hybrid for multi-hop graph-augmented answer generation."
    )
    def ask(
        query: str,
        mode: Literal["vector", "graph", "hybrid"] = "hybrid",
        top_k: int = 3,
        rag_service: RAGService = Depends(get_rag_service)
    ) -> ChatResponse:
        request = ChatRequest(question=query, top_k=top_k, retrieval_mode=mode)
        return rag_service.answer_query(request)

    return app


app = create_application()

