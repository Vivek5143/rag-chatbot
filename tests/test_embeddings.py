from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.embeddings import (
    EmbeddingConfig,
    EmbeddingInputError,
    EmbeddingProviderError,
    EmbeddingService,
    OpenAIEmbeddingProvider,
)


class FakeEmbeddingProvider:
    def __init__(self, *, dimension: int = 3, fail: bool = False) -> None:
        self.dimension = dimension
        self.fail = fail
        self.calls: list[list[str]] = []
        self.requested_dimensions: list[int] = []

    def embed(self, *, texts: list[str], model: str, dimensions: int) -> list[list[float]]:
        del model
        self.calls.append(list(texts))
        self.requested_dimensions.append(dimensions)
        if self.fail:
            raise RuntimeError("provider unavailable")
        return [[float(position)] * self.dimension for position, _ in enumerate(texts)]


def test_embedding_service_rejects_empty_text_without_calling_provider() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider, EmbeddingConfig(model="test", dimension=3))

    with pytest.raises(EmbeddingInputError, match="must not contain empty"):
        service.embed_texts(["valid text", "   "])

    assert provider.calls == []


def test_embedding_service_batches_inputs_in_order() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(
        provider,
        EmbeddingConfig(model="text-embedding-3-small", dimension=3, batch_size=2),
    )

    vectors = service.embed_texts(["one", "two", "three", "four", "five"])

    assert provider.calls == [["one", "two"], ["three", "four"], ["five"]]
    assert provider.requested_dimensions == [3, 3, 3]
    assert len(vectors) == 5
    assert all(len(vector) == 3 for vector in vectors)


def test_embedding_provider_failure_is_explicit() -> None:
    service = EmbeddingService(
        FakeEmbeddingProvider(fail=True),
        EmbeddingConfig(model="text-embedding-3-small", dimension=3),
    )

    with pytest.raises(EmbeddingProviderError, match="Embedding provider request failed"):
        service.embed_texts(["provider failure fixture"])


def test_openai_adapter_uses_configured_model_inputs_and_dimension() -> None:
    endpoint = FakeOpenAIEndpoint(
        response=SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0, 2.0, 2.0]),
                SimpleNamespace(index=0, embedding=[1.0, 1.0, 1.0]),
            ]
        )
    )
    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._client = SimpleNamespace(embeddings=endpoint)

    vectors = provider.embed(
        texts=["first", "second"], model="text-embedding-3-small", dimensions=3
    )

    assert endpoint.calls == [
        {
            "model": "text-embedding-3-small",
            "input": ["first", "second"],
            "dimensions": 3,
        }
    ]
    assert vectors == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


def test_openai_adapter_wraps_provider_failure() -> None:
    provider = object.__new__(OpenAIEmbeddingProvider)
    provider._client = SimpleNamespace(embeddings=FakeOpenAIEndpoint(error=RuntimeError("down")))

    with pytest.raises(EmbeddingProviderError, match="OpenAI embedding request failed"):
        provider.embed(texts=["test"], model="text-embedding-3-small", dimensions=3)


def test_embedding_dimension_mismatch_is_rejected() -> None:
    service = EmbeddingService(
        FakeEmbeddingProvider(dimension=2),
        EmbeddingConfig(model="text-embedding-3-small", dimension=3),
    )

    with pytest.raises(EmbeddingProviderError, match="dimension"):
        service.embed_texts(["wrong vector dimension"])


class FakeOpenAIEndpoint:
    def __init__(self, *, response: object | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response
