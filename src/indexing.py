"""Explicit OpenAI-to-Pinecone indexing command and reusable indexing service.

This module has no FastAPI dependency and is never invoked on application
startup. It validates an existing Pinecone index; provisioning and deletion
remain deliberate operator actions outside this repository's serving process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any, Mapping, Protocol, Sequence
import sys

from langchain_core.documents import Document

from src.config import ConfigurationError, Settings, get_settings
from src.embeddings import (
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingService,
    OpenAIEmbeddingProvider,
)
from src.ingestion import CANONICAL_PDF_PATH, IngestionError, ingest_pdf


# Phase 2's default chunks are 1,000 characters. Keeping that evidence once in
# vector metadata enables the later retrieval phase without a second document
# store, while an explicit bound avoids unexpectedly oversized records.
MAX_METADATA_CONTENT_BYTES = 8_000


class IndexingError(RuntimeError):
    """Base class for explicit indexing failures."""


class IndexingInputError(IndexingError, ValueError):
    """Raised when chunks cannot safely be turned into vector records."""


class PineconeConfigurationError(IndexingError):
    """Raised when the existing index is not compatible with this pipeline."""


class PineconeProviderError(IndexingError):
    """Raised when Pinecone cannot describe or upsert to the configured index."""


@dataclass(frozen=True)
class IndexDescription:
    """The compatibility information required before upserting vectors."""

    name: str
    dimension: int
    metric: str
    host: str
    ready: bool


@dataclass(frozen=True)
class VectorRecord:
    """Provider-neutral vector payload produced from one ingestion chunk."""

    vector_id: str
    values: list[float]
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class IndexingStats:
    """Useful, non-secret result of one explicit indexing invocation."""

    input_chunks: int
    skipped_empty_chunks: int
    embedded_vectors: int
    upserted_vectors: int
    upsert_batches: int


class VectorIndex(Protocol):
    """Small Pinecone-shaped port consumed by :class:`IndexingService`."""

    def describe(self) -> IndexDescription:
        """Describe the pre-provisioned target index."""

    def upsert(self, *, vectors: Sequence[VectorRecord], namespace: str) -> None:
        """Upsert one bounded batch into a namespace."""


class PineconeVectorIndex:
    """Adapter for the official Pinecone SDK's control and data-plane clients."""

    def __init__(self, *, client: Any, index_name: str) -> None:
        self._client = client
        self._index_name = index_name
        self._host: str | None = None
        self._data_index: Any | None = None

    @classmethod
    def from_api_key(cls, *, api_key: str, index_name: str) -> "PineconeVectorIndex":
        if not api_key.strip():
            raise PineconeConfigurationError("Pinecone API key must not be blank")
        if not index_name.strip():
            raise PineconeConfigurationError("Pinecone index name must not be blank")
        try:
            from pinecone import Pinecone
        except ImportError as error:  # pragma: no cover - exercised by installation checks.
            raise PineconeProviderError(
                "The 'pinecone' package is required for live indexing requests"
            ) from error
        return cls(client=Pinecone(api_key=api_key), index_name=index_name)

    def describe(self) -> IndexDescription:
        try:
            response = self._client.describe_index(name=self._index_name)
            status = _response_value(response, "status")
            description = IndexDescription(
                name=str(_required_response_value(response, "name")),
                dimension=int(_required_response_value(response, "dimension")),
                metric=str(_required_response_value(response, "metric")),
                host=str(_required_response_value(response, "host")),
                ready=bool(_required_response_value(status, "ready")),
            )
        except PineconeProviderError:
            raise
        except Exception as error:
            raise PineconeProviderError(
                f"Unable to describe configured Pinecone index '{self._index_name}'"
            ) from error

        self._host = description.host
        return description

    def upsert(self, *, vectors: Sequence[VectorRecord], namespace: str) -> None:
        if not vectors:
            return
        try:
            if self._data_index is None:
                if self._host is None:
                    self.describe()
                self._data_index = self._client.Index(host=self._host)
            payload = [
                {
                    "id": record.vector_id,
                    "values": record.values,
                    "metadata": record.metadata,
                }
                for record in vectors
            ]
            self._data_index.upsert(vectors=payload, namespace=namespace)
        except PineconeProviderError:
            raise
        except Exception as error:
            raise PineconeProviderError(
                f"Pinecone upsert failed for configured index '{self._index_name}'"
            ) from error

    def query(
        self,
        *,
        vector: Sequence[float],
        top_k: int,
        namespace: str,
    ) -> list[Any]:
        """Query the configured index without mutating its contents."""

        if not vector:
            raise PineconeProviderError("Pinecone query vector must not be empty")
        if top_k <= 0:
            raise PineconeProviderError("Pinecone top_k must be greater than zero")
        if not namespace.strip():
            raise PineconeProviderError("Pinecone namespace must not be blank")
        try:
            if self._data_index is None:
                if self._host is None:
                    self.describe()
                self._data_index = self._client.Index(host=self._host)
            response = self._data_index.query(
                vector=list(vector),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
            matches = _response_value(response, "matches")
            if matches is None:
                raise PineconeProviderError("Pinecone query response is missing matches")
            return list(matches)
        except PineconeProviderError:
            raise
        except Exception as error:
            raise PineconeProviderError(
                f"Pinecone query failed for configured index '{self._index_name}'"
            ) from error


class IndexingService:
    """Convert Phase 2 chunks into idempotent vector upserts."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingService,
        vector_index: VectorIndex,
        namespace: str,
        upsert_batch_size: int,
    ) -> None:
        if not namespace.strip():
            raise PineconeConfigurationError("Pinecone namespace must not be blank")
        if upsert_batch_size <= 0:
            raise IndexingInputError("Pinecone upsert batch size must be greater than zero")
        self._embeddings = embeddings
        self._vector_index = vector_index
        self._namespace = namespace
        self._upsert_batch_size = upsert_batch_size

    def index_chunks(self, chunks: Sequence[Document]) -> IndexingStats:
        """Validate index/chunks, embed content, and upsert stable records in batches."""

        input_chunks = len(chunks)
        non_empty_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        skipped_empty = input_chunks - len(non_empty_chunks)
        if not non_empty_chunks:
            raise IndexingInputError("No non-empty chunks are available for indexing")

        self._validate_index_compatibility(self._vector_index.describe())
        self._validate_unique_chunk_ids(non_empty_chunks)

        texts = [chunk.page_content for chunk in non_empty_chunks]
        vectors = self._embeddings.embed_texts(texts)
        if len(vectors) != len(non_empty_chunks):
            raise IndexingError("Embedding service returned an unexpected vector count")

        records = [
            _vector_record(chunk=chunk, vector=vector)
            for chunk, vector in zip(non_empty_chunks, vectors, strict=True)
        ]
        batches = list(_batches(records, self._upsert_batch_size))
        for batch in batches:
            self._vector_index.upsert(vectors=batch, namespace=self._namespace)

        return IndexingStats(
            input_chunks=input_chunks,
            skipped_empty_chunks=skipped_empty,
            embedded_vectors=len(records),
            upserted_vectors=len(records),
            upsert_batches=len(batches),
        )

    def _validate_index_compatibility(self, description: IndexDescription) -> None:
        if not description.ready:
            raise PineconeConfigurationError(
                f"Pinecone index '{description.name}' is not ready"
            )
        if description.metric.lower() != "cosine":
            raise PineconeConfigurationError(
                f"Pinecone index '{description.name}' must use cosine metric"
            )
        if description.dimension != self._embeddings.config.dimension:
            raise PineconeConfigurationError(
                f"Pinecone index '{description.name}' has dimension {description.dimension}; "
                f"expected {self._embeddings.config.dimension}"
            )

    @staticmethod
    def _validate_unique_chunk_ids(chunks: Sequence[Document]) -> None:
        chunk_ids = [str(chunk.metadata.get("chunk_id", "")) for chunk in chunks]
        if any(not chunk_id for chunk_id in chunk_ids):
            raise IndexingInputError("Every chunk must include a non-empty deterministic chunk_id")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise IndexingInputError("Chunks for one indexing run must have unique chunk_id values")


def index_canonical_pdf(settings: Settings | None = None) -> IndexingStats:
    """Run the explicit canonical-PDF indexing workflow, never from FastAPI."""

    active_settings = settings or get_settings()
    # This comes first so an operator is told about the absent approved source
    # even if their local credentials have not been supplied yet.
    chunks = ingest_pdf(
        CANONICAL_PDF_PATH,
        chunk_size=active_settings.chunk_size,
        chunk_overlap=active_settings.chunk_overlap,
    )
    active_settings.require_indexing_configuration()

    assert active_settings.openai_api_key is not None
    assert active_settings.pinecone_api_key is not None
    embeddings = EmbeddingService(
        OpenAIEmbeddingProvider(
            active_settings.openai_api_key.get_secret_value(),
            base_url=active_settings.openai_base_url,
        ),
        EmbeddingConfig(
            model=active_settings.openai_embedding_model,
            dimension=active_settings.embedding_dimension,
            batch_size=active_settings.embedding_batch_size,
        ),
    )
    vector_index = PineconeVectorIndex.from_api_key(
        api_key=active_settings.pinecone_api_key.get_secret_value(),
        index_name=active_settings.pinecone_index_name,
    )
    indexer = IndexingService(
        embeddings=embeddings,
        vector_index=vector_index,
        namespace=active_settings.pinecone_namespace,
        upsert_batch_size=active_settings.pinecone_upsert_batch_size,
    )
    return indexer.index_chunks(chunks)


def main() -> int:
    """Run indexing as ``python -m src.indexing`` and report concise statistics."""

    try:
        settings = get_settings()
        stats = index_canonical_pdf(settings)
    except (ConfigurationError, EmbeddingError, IngestionError, IndexingError, ValueError) as error:
        print(f"Indexing failed: {error}", file=sys.stderr)
        return 1

    print("Indexing completed")
    print(f"index: {settings.pinecone_index_name}")
    print(f"namespace: {settings.pinecone_namespace}")
    print(f"embedding_model: {settings.openai_embedding_model}")
    for key, value in asdict(stats).items():
        print(f"{key}: {value}")
    return 0


def _vector_record(*, chunk: Document, vector: list[float]) -> VectorRecord:
    metadata = chunk.metadata
    required = ("chunk_id", "source", "page", "document_sha256")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise IndexingInputError(
            "Chunk is missing required provenance metadata: " + ", ".join(missing)
        )

    chunk_id = str(metadata["chunk_id"])
    source = str(metadata["source"])
    source_hash = str(metadata["document_sha256"])
    page = metadata["page"]
    content = chunk.page_content
    if not chunk_id or not source or not source_hash:
        raise IndexingInputError("Chunk provenance metadata must not be blank")
    if not isinstance(page, int) or page < 1:
        raise IndexingInputError("Chunk page metadata must be a positive integer")
    if len(content.encode("utf-8")) > MAX_METADATA_CONTENT_BYTES:
        raise IndexingInputError(
            "Chunk content exceeds the configured Pinecone metadata evidence limit"
        )

    return VectorRecord(
        vector_id=chunk_id,
        values=vector,
        metadata={
            "chunk_id": chunk_id,
            "source": source,
            "page": page,
            "source_hash": source_hash,
            "content": content,
        },
    )


def _batches(records: Sequence[VectorRecord], size: int) -> list[list[VectorRecord]]:
    iterator = iter(records)
    return [batch for batch in iter(lambda: list(islice(iterator, size)), [])]


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(key)
    return getattr(response, key, None)


def _required_response_value(response: Any, key: str) -> Any:
    value = _response_value(response, key)
    if value is None or value == "":
        raise PineconeProviderError(f"Pinecone index description is missing '{key}'")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
