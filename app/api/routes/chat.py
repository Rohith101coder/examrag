"""Chat and Question-Answering endpoint."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_rag_service, verify_chat_access
from app.api.schemas.chat import ChatRequest, ChatResponse
from app.core.exceptions import (
    CollectionDimensionMismatchError,
    ProviderError,
    ProviderTimeoutError,
    StructuredOutputParsingError,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post(
    "/query",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Query indexed course documents",
    description="Accepts a question and optional document IDs, retrieves grounded passages, and generates an exam-ready answer with source citations."
)
def query_exam_rag(
    request: ChatRequest,
    api_key: Optional[str] = Depends(verify_chat_access),
    rag_service: RAGService = Depends(get_rag_service)
) -> ChatResponse:
    try:
        return rag_service.answer_query(request)
    except ProviderTimeoutError as e:
        logger.error("Provider timeout in chat query: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The external language model provider timed out while generating the answer."
        )
    except CollectionDimensionMismatchError as e:
        logger.error("Collection dimension mismatch in chat query: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message
        )
    except StructuredOutputParsingError as e:
        logger.error("Failed to parse structured output: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The model provider returned an unparseable response structure. Please retry."
        )
    except ProviderError as e:
        logger.error("Provider error in chat query: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="An error occurred communicating with the AI model provider."
        )
    except Exception as e:
        logger.error("Unexpected error in chat query: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing your request."
        )

