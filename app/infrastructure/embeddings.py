"""OpenRouter-compatible Embeddings infrastructure using LangChain."""

import logging
from typing import List, Optional

import httpx
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, ProviderError, ProviderTimeoutError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around OpenAI-compatible embeddings via OpenRouter."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._dimension: Optional[int] = None
        self._client: Optional[OpenAIEmbeddings] = None

    def _get_client(self) -> OpenAIEmbeddings:
        if self._client is None:
            if not self.settings.OPENROUTER_API_KEY:
                raise ConfigurationError("OPENROUTER_API_KEY is not configured.")

            self._client = OpenAIEmbeddings(
                api_key=self.settings.OPENROUTER_API_KEY,
                base_url=self.settings.OPENROUTER_BASE_URL,
                model=self.settings.EMBEDDING_MODEL,
                timeout=self.settings.LLM_TIMEOUT_SECONDS,
                max_retries=self.settings.LLM_MAX_RETRIES,
                check_embedding_ctx_length=False
            )
        return self._client

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding vector for a single search query."""
        client = self._get_client()
        try:
            vector = client.embed_query(text)
            if self._dimension is None and vector:
                self._dimension = len(vector)
            return vector
        except (httpx.TimeoutException, TimeoutError) as e:
            logger.error("Embedding query timed out: %s", str(e))
            raise ProviderTimeoutError(f"Embedding service timed out after {self.settings.LLM_TIMEOUT_SECONDS}s") from e
        except Exception as e:
            logger.error("Error generating query embedding: %s", str(e))
            raise ProviderError(f"Failed to generate query embedding: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of document chunks in batches."""
        if not texts:
            return []

        client = self._get_client()
        batch_size = self.settings.EMBEDDING_BATCH_SIZE
        all_embeddings: List[List[float]] = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                if self._dimension is None and batch_embeddings:
                    self._dimension = len(batch_embeddings[0])

            return all_embeddings
        except (httpx.TimeoutException, TimeoutError) as e:
            logger.error("Embedding documents timed out: %s", str(e))
            raise ProviderTimeoutError(f"Embedding service timed out after {self.settings.LLM_TIMEOUT_SECONDS}s") from e
        except Exception as e:
            logger.error("Error generating document embeddings: %s", str(e))
            raise ProviderError(f"Failed to generate document embeddings: {str(e)}") from e

    def get_dimension(self) -> int:
        """Dynamically detect or return cached embedding dimension."""
        if self._dimension is not None:
            return self._dimension

        # Known defaults for common models
        if "text-embedding-3-small" in self.settings.EMBEDDING_MODEL:
            self._dimension = 1536
            return 1536
        if "text-embedding-3-large" in self.settings.EMBEDDING_MODEL:
            self._dimension = 3072
            return 3072

        # Probe with a short token
        vector = self.embed_query("examrag_dimension_probe")
        self._dimension = len(vector)
        return self._dimension

