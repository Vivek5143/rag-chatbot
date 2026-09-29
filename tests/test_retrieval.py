from __future__ import annotations

from typing import Any

import pytest

from src.embeddings import EmbeddingConfig, EmbeddingError, EmbeddingService
from src.models import RetrievedCandidate
from src.retrieval import (
    RetrievalCandidateMappingError,
    RetrievalEmbeddingError,
    RetrievalInputError,
    RetrievalProviderError,
    RetrievalService,
)


class FakeEmbeddingProvider:
    def __init__(self, *, dimension: int = 3, fail: bool = False) -> None:
        self.dimension = dimension
        self.fail = fail
        self.calls: list[list[str]] = []
        self.requested_models: list[str] = []
        self.requested_dimensions: list[int] = []

    def embed(
        self, *, texts: list[str], model: str, dimensions: int
    ) -> list[list[float]]:
        del model
        self.calls.append(list(texts))
        self.requested_dimensions.append(dimensions)
        if self.fail:
            raise RuntimeError("embedding provider unavailable")
        return [[float(idx)] * self.dimension for idx, _ in enumerate(texts)]


class FakeVectorIndex:
    def __init__(
        self, *, matches: list[dict[str, Any]] | None = None, fail: bool = False
    ) -> None:
        self.matches = matches or []
        self.fail = fail
        self.query_calls: list[dict[str, Any]] = []

    def query(
        self, *, vector: list[float], top_k: int, namespace: str
    ) -> list[Any]:
        self.query_calls.append(
            {"vector": list(vector), "top_k": top_k, "namespace": namespace}
        )
        if self.fail:
            raise RuntimeError("pinecone unavailable")
        return list(self.matches)


def _embedding_service(*, dimension: int = 3, batch_size: int = 64, fail: bool = False) -> EmbeddingService:
    provider = FakeEmbeddingProvider(dimension=dimension, fail=fail)
    return EmbeddingService(
        provider,  # type: ignore[arg-type]
        EmbeddingConfig(model="test-embedding-model", dimension=dimension, batch_size=batch_size),
    )


def _candidate_match(
    *, chunk_id: str = "c1", content: str = "hello", page: int = 1
) -> dict[str, Any]:
    return {
        "id": chunk_id,
        "score": 0.77,
        "metadata": {
            "chunk_id": chunk_id,
            "content": content,
            "source": "Ebook-Agentic-AI.pdf",
            "page": page,
            "source_hash": "a" * 64,
        },
    }


def test_retrieve_empty_question_rejects_without_calling_embedding() -> None:
    provider = FakeEmbeddingProvider()
    embeddings = EmbeddingService(
        provider, EmbeddingConfig(model="test-embedding-model", dimension=3)
    )
    index = FakeVectorIndex(matches=[_candidate_match()])
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    with pytest.raises(RetrievalInputError, match="question"):
        service.retrieve(question="   ")

    assert provider.calls == []
    assert index.query_calls == []


def test_retrieve_embeds_stripped_question_and_uses_top_k_and_namespace() -> None:
    provider = FakeEmbeddingProvider(dimension=3)
    embeddings = EmbeddingService(
        provider,
        EmbeddingConfig(
            model="test-embedding-model", dimension=3, batch_size=2
        ),
    )
    index = FakeVectorIndex(matches=[_candidate_match()])

    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    _ = service.retrieve(question="  What is Agentic AI?  ")

    assert provider.calls == [["What is Agentic AI?"]]
    assert provider.requested_dimensions == [3]
    assert index.query_calls[0]["top_k"] == 4
    assert index.query_calls[0]["namespace"] == "ebook-v1"


def test_retrieve_preserves_similarity_score_and_metadata() -> None:
    index = FakeVectorIndex(matches=[_candidate_match(chunk_id="chunk-1", content="Chunk text")])
    embeddings = _embedding_service(dimension=3)
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    candidates = service.retrieve(question="q")

    assert len(candidates) == 1
    candidate = candidates[0]
    assert isinstance(candidate, RetrievedCandidate)

    assert candidate.chunk_id == "chunk-1"
    assert candidate.content == "Chunk text"
    assert candidate.source == "Ebook-Agentic-AI.pdf"
    assert candidate.page == 1
    assert candidate.source_hash == "a" * 64
    assert candidate.similarity_score == 0.77
    assert candidate.metadata["source"] == "Ebook-Agentic-AI.pdf"


def test_retrieve_raises_on_missing_required_metadata() -> None:
    bad_match = {
        "id": "chunk-1",
        "score": 0.77,
        "metadata": {
            # missing: content
            "chunk_id": "chunk-1",
            "source": "Ebook-Agentic-AI.pdf",
            "page": 1,
            "source_hash": "a" * 64,
        },
    }
    index = FakeVectorIndex(matches=[bad_match])
    embeddings = _embedding_service(dimension=3)
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    with pytest.raises(RetrievalCandidateMappingError, match="content"):
        service.retrieve(question="q")


def test_retrieve_propagates_pinecone_failure_as_typed_error() -> None:
    index = FakeVectorIndex(matches=[_candidate_match()], fail=True)
    embeddings = _embedding_service(dimension=3)
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    with pytest.raises(RetrievalProviderError, match="pinecone"):
        service.retrieve(question="q")


def test_retrieve_propagates_embedding_failure_as_typed_error() -> None:
    embeddings = _embedding_service(dimension=3, fail=True)
    index = FakeVectorIndex(matches=[_candidate_match()])
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    with pytest.raises(RetrievalEmbeddingError, match="embedding"):
        service.retrieve(question="q")


def test_retrieve_returns_empty_list_when_pinecone_returns_no_matches() -> None:
    index = FakeVectorIndex(matches=[])
    embeddings = _embedding_service(dimension=3)
    service = RetrievalService(
        embeddings=embeddings,
        vector_index=index,
        namespace="ebook-v1",
        top_k=4,
    )

    candidates = service.retrieve(question="q")
    assert candidates == []
