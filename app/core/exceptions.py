"""Custom domain and application exceptions for ExamRAG."""


class ExamRAGException(Exception):
    """Base exception for ExamRAG application errors."""

    def __init__(self, message: str, detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.detail = detail or message


class ConfigurationError(ExamRAGException):
    """Raised when environment or startup configuration is invalid."""
    pass


class InvalidPDFError(ExamRAGException):
    """Raised when an uploaded file is not a valid PDF or has invalid magic bytes."""
    pass


class DocumentParsingError(ExamRAGException):
    """Raised when PDF text extraction fails or the document is empty/unreadable."""
    pass


class OversizedFileError(ExamRAGException):
    """Raised when an uploaded document exceeds the configured size limit."""
    pass


class CollectionDimensionMismatchError(ExamRAGException):
    """Raised when the embedding dimension does not match existing Qdrant collection."""
    pass


class VectorStoreError(ExamRAGException):
    """Raised when an error occurs during Qdrant operations."""
    pass


class ProviderError(ExamRAGException):
    """Raised when an external model provider (OpenRouter) returns an error."""
    pass


class ProviderTimeoutError(ProviderError):
    """Raised when a request to an external provider times out."""
    pass


class DocumentNotFoundError(ExamRAGException):
    """Raised when a requested document ID is not found in the index."""
    pass


class AuthenticationError(ExamRAGException):
    """Raised when an API key is missing or invalid."""
    pass


class IngestionDisabledError(ExamRAGException):
    """Raised when local filesystem ingestion is disabled in production."""
    pass


class StructuredOutputParsingError(ExamRAGException):
    """Raised when LLM output cannot be parsed into the expected exam answer schema."""
    pass

