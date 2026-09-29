from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.embeddings import EmbeddingError, EmbeddingService
from src.indexing import PineconeVectorIndex
from src.models import RetrievedCandidate


class RetrievalError(RuntimeError):
    """Base class for retrieval failures."""


class RetrievalInputError(RetrievalError, ValueError):
    """Raised when retrieval inputs are malformed."""


class RetrievalEmbeddingError(RetrievalError):
    """Raised when the embedding stage fails."""


class RetrievalProviderError(RetrievalError):
    """Raised when Pinecone cannot answer the query."""


class RetrievalCandidateMappingError(RetrievalError):
    """Raised when a Pinecone match cannot be mapped to a candidate."""


def _response_value(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


@dataclass(frozen=True)
class RetrievalService:
    """Retrieve provenance-rich candidates from an existing Pinecone index."""

    embeddings: EmbeddingService
    vector_index: PineconeVectorIndex
    namespace: str
    top_k: int

    def __post_init__(self) -> None:
        if not self.namespace.strip():
            raise RetrievalInputError("namespace must not be blank")
        if self.top_k < 1:
            raise RetrievalInputError("top_k must be greater than zero")

    def retrieve(self, *, question: str) -> list[RetrievedCandidate]:
        normalized = question.strip()
        if not normalized:
            raise RetrievalInputError("question must not be blank")

        try:
            vectors = self.embeddings.embed_texts([normalized])
        except EmbeddingError as error:
            raise RetrievalEmbeddingError("embedding failed") from error
        except Exception as error:
            raise RetrievalEmbeddingError("embedding failed") from error

        if not vectors or not vectors[0]:
            raise RetrievalEmbeddingError("embedding returned an empty vector")

        try:
            matches = self.vector_index.query(
                vector=vectors[0],
                top_k=self.top_k,
                namespace=self.namespace,
            )
        except Exception as error:
            raise RetrievalProviderError("pinecone query failed") from error

        return [self._map_match(match) for match in matches]

    @staticmethod
    def _required_metadata_fields() -> Sequence[str]:
        return ("content", "source", "page", "source_hash")

    def _map_match(self, match: Any) -> RetrievedCandidate:
        metadata = _response_value(match, "metadata")
        if not isinstance(metadata, Mapping):
            raise RetrievalCandidateMappingError("match metadata is missing or invalid")

        chunk_id_from_metadata = metadata.get("chunk_id")
        chunk_id_from_match = _response_value(match, "id")
        chunk_id = str(chunk_id_from_metadata or chunk_id_from_match or "")
        if not chunk_id:
            raise RetrievalCandidateMappingError("chunk_id is missing")

        missing = [
            key
            for key in self._required_metadata_fields()
            if key not in metadata
        ]
        if missing:
            raise RetrievalCandidateMappingError(
                "match metadata is missing required fields: " + ", ".join(missing)
            )

        try:
            page = int(metadata["page"])
        except Exception as error:
            raise RetrievalCandidateMappingError("page must be an integer") from error

        score = _response_value(match, "score")
        if score is None:
            raise RetrievalCandidateMappingError("match score is missing")

        content = str(metadata["content"])
        source = str(metadata["source"])
        source_hash = str(metadata["source_hash"])

        return RetrievedCandidate(
            chunk_id=chunk_id,
            content=content,
            source=source,
            page=page,
            source_hash=source_hash,
            similarity_score=float(score),
            metadata=dict(metadata),
        )
