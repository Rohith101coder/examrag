"""Qdrant Cloud vector store infrastructure and collection management."""

import logging
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    CollectionDimensionMismatchError,
    ConfigurationError,
    VectorStoreError,
)

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Production wrapper for Qdrant Cloud vector store operations."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[QdrantClient] = None):
        self.settings = settings or get_settings()
        self._client = client

    def get_client(self) -> QdrantClient:
        """Initialize or return cached QdrantClient instance."""
        if self._client is None:
            if not self.settings.QDRANT_URL or not self.settings.QDRANT_API_KEY:
                raise ConfigurationError("QDRANT_URL and QDRANT_API_KEY must be configured.")

            try:
                self._client = QdrantClient(
                    url=self.settings.QDRANT_URL,
                    api_key=self.settings.QDRANT_API_KEY,
                    timeout=self.settings.LLM_TIMEOUT_SECONDS,
                    check_compatibility=False
                )
            except Exception as e:
                logger.error("Failed to initialize Qdrant client: %s", str(e))
                raise VectorStoreError(f"Failed to connect to Qdrant Cloud: {str(e)}") from e

        return self._client

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if missing or validate dimension if it already exists."""
        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        try:
            exists = client.collection_exists(collection_name)
            if not exists:
                logger.info(
                    "Creating Qdrant collection '%s' with dimension %d and Cosine distance.",
                    collection_name,
                    dimension
                )
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=models.Distance.COSINE
                    )
                )

                # Create payload indexes for frequent filtering
                self._create_payload_indexes(client, collection_name)
            else:
                # Validate dimension of existing collection
                info = client.get_collection(collection_name)
                vectors_config = info.config.params.vectors

                # vectors_config can be VectorParams or dict of VectorParams
                if isinstance(vectors_config, models.VectorParams):
                    existing_dim = vectors_config.size
                elif isinstance(vectors_config, dict):
                    existing_dim = next(iter(vectors_config.values())).size
                else:
                    existing_dim = getattr(vectors_config, "size", dimension)

                if existing_dim != dimension:
                    raise CollectionDimensionMismatchError(
                        f"Qdrant collection '{collection_name}' dimension mismatch: "
                        f"collection has {existing_dim}, but embedding model requires {dimension}."
                    )
                logger.debug("Qdrant collection '%s' verified (dim=%d).", collection_name, dimension)

        except CollectionDimensionMismatchError:
            raise
        except Exception as e:
            logger.error("Failed to verify or create Qdrant collection '%s': %s", collection_name, str(e))
            raise VectorStoreError(f"Vector store collection check failed: {str(e)}") from e

    def _create_payload_indexes(self, client: QdrantClient, collection_name: str) -> None:
        """Create payload indexes for fast filtering on document_id and content_hash."""
        for field in ["document_id", "content_hash"]:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                logger.debug("Created index for field '%s' on collection '%s'", field, collection_name)
            except Exception as e:
                # Non-fatal if index creation fails or already exists
                logger.warning("Could not create payload index for '%s': %s", field, str(e))

    def upsert_points(self, points: List[models.PointStruct]) -> None:
        """Upsert a list of PointStruct vectors and payloads into Qdrant."""
        if not points:
            return

        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        try:
            # Batch upserts in chunks of 100 points
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            logger.info("Upserted %d points into '%s'", len(points), collection_name)
        except Exception as e:
            logger.error("Failed to upsert points into Qdrant: %s", str(e))
            raise VectorStoreError(f"Failed to persist vectors in Qdrant: {str(e)}") from e

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        document_ids: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[models.ScoredPoint]:
        """Perform cosine similarity search on the collection with optional document filtering."""
        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        query_filter: Optional[models.Filter] = None
        if document_ids:
            if len(document_ids) == 1:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_ids[0]))]
                )
            else:
                query_filter = models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids))]
                )

        try:
            # Use query_points or search depending on client version
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True
                )
                return response.points
            else:
                return client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True
                )
        except UnexpectedResponse as e:
            logger.error("Qdrant search unexpected response: %s", str(e))
            raise VectorStoreError(f"Qdrant query failed: {str(e)}") from e
        except Exception as e:
            logger.error("Error during Qdrant search: %s", str(e))
            raise VectorStoreError(f"Vector search failed: {str(e)}") from e

    def delete_by_document_id(self, document_id: str) -> int:
        """Delete all points belonging to a specific document_id."""
        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        try:
            # Count points first to report how many were deleted
            count_res = client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            )
            deleted_count = count_res.count

            if deleted_count > 0:
                client.delete(
                    collection_name=collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                        )
                    ),
                    wait=True
                )
                logger.info("Deleted %d points for document_id '%s'", deleted_count, document_id)

            return deleted_count
        except Exception as e:
            logger.error("Failed to delete points for document_id '%s': %s", document_id, str(e))
            raise VectorStoreError(f"Failed to delete document from vector store: {str(e)}") from e

    def get_document_metadata(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve aggregated metadata and chunk count for a document."""
        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        try:
            if not client.collection_exists(collection_name):
                return None

            count_res = client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            )
            chunk_count = count_res.count
            if chunk_count == 0:
                return None

            # Fetch one sample point to read filename and content_hash
            records, _ = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                ),
                limit=1,
                with_payload=True
            )

            if not records:
                return None

            payload = records[0].payload or {}
            return {
                "document_id": document_id,
                "filename": payload.get("filename", "unknown.pdf"),
                "content_hash": payload.get("content_hash", ""),
                "chunk_count": chunk_count
            }
        except Exception as e:
            logger.error("Failed to retrieve document metadata for '%s': %s", document_id, str(e))
            raise VectorStoreError(f"Failed to get document metadata: {str(e)}") from e

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List unique indexed documents by aggregating payload metadata from Qdrant."""
        client = self.get_client()
        collection_name = self.settings.QDRANT_COLLECTION_NAME

        try:
            if not client.collection_exists(collection_name):
                return []

            # Scroll through points to aggregate unique documents
            documents: Dict[str, Dict[str, Any]] = {}
            offset = None

            while True:
                records, offset = client.scroll(
                    collection_name=collection_name,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for record in records:
                    payload = record.payload or {}
                    doc_id = payload.get("document_id")
                    if not doc_id:
                        continue

                    if doc_id not in documents:
                        documents[doc_id] = {
                            "document_id": doc_id,
                            "filename": payload.get("filename", "unknown.pdf"),
                            "content_hash": payload.get("content_hash", ""),
                            "chunk_count": 0
                        }
                    documents[doc_id]["chunk_count"] += 1

                if offset is None or len(documents) >= limit:
                    break

            return list(documents.values())[:limit]
        except Exception as e:
            logger.error("Failed to list indexed documents: %s", str(e))
            raise VectorStoreError(f"Failed to list documents: {str(e)}") from e

    def health_check(self) -> bool:
        """Check whether Qdrant cluster is reachable."""
        try:
            client = self.get_client()
            client.get_collections()
            return True
        except Exception as e:
            logger.warning("Qdrant health check failed: %s", str(e))
            return False
