from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.embeddings import EmbeddingConfig, EmbeddingProviderError, EmbeddingService
from src.indexing import (
    IndexDescription,
    IndexingInputError,
    IndexingService,
    PineconeConfigurationError,
    PineconeVectorIndex,
    VectorRecord,
)


class FakeEmbeddingProvider:
    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, *, texts: list[str], model: str, dimensions: int) -> list[list[float]]:
        del model, dimensions
        self.calls.append(list(texts))
        return [[float(index + 1)] * self.dimension for index, _ in enumerate(texts)]


class FakeVectorIndex:
    def __init__(self, description: IndexDescription | None = None) -> None:
        self.description = description or IndexDescription(
            name="agentic-ai-index",
            dimension=3,
            metric="cosine",
            host="test-host",
            ready=True,
        )
        self.upserts: list[tuple[str, list[VectorRecord]]] = []

    def describe(self) -> IndexDescription:
        return self.description

    def upsert(self, *, vectors: list[VectorRecord], namespace: str) -> None:
        self.upserts.append((namespace, list(vectors)))


def test_indexing_preserves_provenance_metadata_namespace_and_batches() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex()
    service = _indexing_service(provider, index, batch_size=2)

    stats = service.index_chunks(_chunks(5))

    assert stats.upserted_vectors == 5
    assert stats.upsert_batches == 3
    assert [len(vectors) for _, vectors in index.upserts] == [2, 2, 1]
    assert [namespace for namespace, _ in index.upserts] == ["ebook-v1"] * 3
    record = index.upserts[0][1][0]
    assert record.vector_id == "chunk-0"
    assert record.metadata == {
        "chunk_id": "chunk-0",
        "source": "Ebook-Agentic-AI.pdf",
        "page": 1,
        "source_hash": "a" * 64,
        "content": "Chunk content 0",
    }


def test_repeated_ingestion_reuses_the_same_vector_ids() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex()
    service = _indexing_service(provider, index, batch_size=10)
    chunks = _chunks(3)

    first = service.index_chunks(chunks)
    second = service.index_chunks(chunks)

    first_ids = [record.vector_id for record in index.upserts[0][1]]
    second_ids = [record.vector_id for record in index.upserts[1][1]]
    assert first.upserted_vectors == second.upserted_vectors == 3
    assert first_ids == second_ids == ["chunk-0", "chunk-1", "chunk-2"]


def test_empty_chunks_are_never_sent_to_embedding_provider() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex()
    service = _indexing_service(provider, index)

    with pytest.raises(IndexingInputError, match="No non-empty chunks"):
        service.index_chunks([_chunk(chunk_id="empty", content="   ")])

    assert provider.calls == []
    assert index.upserts == []


def test_incompatible_pinecone_metric_fails_before_embedding() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex(
        IndexDescription(
            name="agentic-ai-index",
            dimension=3,
            metric="dotproduct",
            host="test-host",
            ready=True,
        )
    )
    service = _indexing_service(provider, index)

    with pytest.raises(PineconeConfigurationError, match="cosine"):
        service.index_chunks(_chunks(1))

    assert provider.calls == []
    assert index.upserts == []


def test_pinecone_dimension_mismatch_fails_before_embedding() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex(
        IndexDescription(
            name="agentic-ai-index",
            dimension=4,
            metric="cosine",
            host="test-host",
            ready=True,
        )
    )
    service = _indexing_service(provider, index)

    with pytest.raises(PineconeConfigurationError, match="dimension 4"):
        service.index_chunks(_chunks(1))

    assert provider.calls == []


def test_embedding_dimension_mismatch_prevents_partial_upsert() -> None:
    provider = FakeEmbeddingProvider(dimension=2)
    index = FakeVectorIndex()
    service = _indexing_service(provider, index)

    with pytest.raises(EmbeddingProviderError, match="dimension"):
        service.index_chunks(_chunks(2))

    assert index.upserts == []


def test_duplicate_chunk_ids_in_one_run_are_rejected() -> None:
    provider = FakeEmbeddingProvider()
    index = FakeVectorIndex()
    service = _indexing_service(provider, index)
    chunks = [_chunk(chunk_id="same", content="First"), _chunk(chunk_id="same", content="Second")]

    with pytest.raises(IndexingInputError, match="unique"):
        service.index_chunks(chunks)

    assert provider.calls == []
    assert index.upserts == []


def test_pinecone_adapter_targets_described_host_and_namespace() -> None:
    client = FakePineconeClient()
    adapter = PineconeVectorIndex(client=client, index_name="agentic-ai-index")
    record = VectorRecord(
        vector_id="chunk-1",
        values=[1.0, 2.0, 3.0],
        metadata={"chunk_id": "chunk-1", "source": "fixture.pdf", "page": 1},
    )

    description = adapter.describe()
    adapter.upsert(vectors=[record], namespace="ebook-v1")

    assert description.dimension == 3
    assert client.describe_calls == ["agentic-ai-index"]
    assert client.index_hosts == ["test-host"]
    assert client.data_index.upserts == [
        (
            "ebook-v1",
            [
                {
                    "id": "chunk-1",
                    "values": [1.0, 2.0, 3.0],
                    "metadata": {"chunk_id": "chunk-1", "source": "fixture.pdf", "page": 1},
                }
            ],
        )
    ]


def _indexing_service(
    provider: FakeEmbeddingProvider,
    index: FakeVectorIndex,
    *,
    batch_size: int = 100,
) -> IndexingService:
    return IndexingService(
        embeddings=EmbeddingService(
            provider,
            EmbeddingConfig(model="text-embedding-3-small", dimension=3, batch_size=2),
        ),
        vector_index=index,
        namespace="ebook-v1",
        upsert_batch_size=batch_size,
    )


def _chunks(count: int) -> list[Document]:
    return [_chunk(chunk_id=f"chunk-{index}", content=f"Chunk content {index}") for index in range(count)]


def _chunk(*, chunk_id: str, content: str) -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_id": chunk_id,
            "source": "Ebook-Agentic-AI.pdf",
            "page": 1,
            "document_sha256": "a" * 64,
        },
    )


class FakePineconeClient:
    def __init__(self) -> None:
        self.describe_calls: list[str] = []
        self.index_hosts: list[str] = []
        self.data_index = FakePineconeDataIndex()

    def describe_index(self, *, name: str) -> dict[str, object]:
        self.describe_calls.append(name)
        return {
            "name": name,
            "dimension": 3,
            "metric": "cosine",
            "host": "test-host",
            "status": {"ready": True},
        }

    def Index(self, *, host: str) -> "FakePineconeDataIndex":
        self.index_hosts.append(host)
        return self.data_index


class FakePineconeDataIndex:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, list[dict[str, object]]]] = []

    def upsert(self, *, vectors: list[dict[str, object]], namespace: str) -> None:
        self.upserts.append((namespace, vectors))
