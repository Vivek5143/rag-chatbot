"""Small, testable adapter for document embeddings.

The module deliberately creates no OpenAI client and makes no provider call at
import time. ``EmbeddingService`` depends on a small protocol so unit tests can
use an offline fake rather than the OpenAI SDK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


DEFAULT_EMBEDDING_DIMENSION = 1536


class EmbeddingError(RuntimeError):
    """Base class for safe, explicit embedding failures."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """Raised before a malformed input can reach an embedding provider."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the provider call or its response is unusable."""


class EmbeddingProvider(Protocol):
    """Minimal provider port used by the batching service."""

    def embed(
        self, *, texts: Sequence[str], model: str, dimensions: int
    ) -> list[list[float]]:
        """Return one vector per input text, in the same order."""


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration that must agree with the target vector index."""

    model: str
    dimension: int = DEFAULT_EMBEDDING_DIMENSION
    batch_size: int = 64

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise EmbeddingInputError("Embedding model must not be blank")
        if self.dimension <= 0:
            raise EmbeddingInputError("Embedding dimension must be greater than zero")
        if self.batch_size <= 0:
            raise EmbeddingInputError("Embedding batch size must be greater than zero")


class OpenAIEmbeddingProvider:
    """Official OpenAI-SDK adapter kept behind :class:`EmbeddingProvider`."""

    def __init__(self, api_key: str, base_url: str = "") -> None:
        if not api_key.strip():
            raise EmbeddingInputError("OpenAI API key must not be blank")

        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - exercised by installation checks.
            raise EmbeddingProviderError(
                "The 'openai' package is required for live embedding requests"
            ) from error

        # Client construction is local to explicit indexing/runtime wiring; it
        # performs no embedding request during module import. ``base_url`` is
        # empty by default, so the client targets the official OpenAI endpoint;
        # an explicit value redirects it at an OpenAI-compatible provider.
        client_kwargs: dict[str, object] = {"api_key": api_key}
        if base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        self._client = OpenAI(**client_kwargs)

    def embed(
        self, *, texts: Sequence[str], model: str, dimensions: int
    ) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=model,
                input=list(texts),
                dimensions=dimensions,
            )
        except Exception as error:  # The SDK exposes multiple transport/API exceptions.
            raise EmbeddingProviderError("OpenAI embedding request failed") from error

        try:
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(texts))):
                raise ValueError("response indices do not map to requested input order")
            vectors = [list(item.embedding) for item in ordered]
        except (AttributeError, TypeError, ValueError) as error:
            raise EmbeddingProviderError("OpenAI returned an invalid embedding response") from error

        if len(vectors) != len(texts):
            raise EmbeddingProviderError(
                "OpenAI returned a different number of embeddings than requested"
            )
        return vectors


class EmbeddingService:
    """Validate, batch, and dimension-check document embeddings."""

    def __init__(self, provider: EmbeddingProvider, config: EmbeddingConfig) -> None:
        self._provider = provider
        self.config = config

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed non-empty texts in stable order, without provider-specific leakage."""

        normalized_texts = list(texts)
        if not normalized_texts:
            raise EmbeddingInputError("At least one text is required for embedding")
        if any(not isinstance(text, str) or not text.strip() for text in normalized_texts):
            raise EmbeddingInputError("Embedding inputs must not contain empty text")

        vectors: list[list[float]] = []
        for start in range(0, len(normalized_texts), self.config.batch_size):
            batch = normalized_texts[start : start + self.config.batch_size]
            try:
                batch_vectors = self._provider.embed(
                    texts=batch,
                    model=self.config.model,
                    dimensions=self.config.dimension,
                )
            except EmbeddingError:
                raise
            except Exception as error:
                raise EmbeddingProviderError("Embedding provider request failed") from error

            if len(batch_vectors) != len(batch):
                raise EmbeddingProviderError(
                    "Embedding provider returned a different number of vectors than requested"
                )
            for vector in batch_vectors:
                self._validate_vector(vector)
                vectors.append(list(vector))

        return vectors

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if len(vector) != self.config.dimension:
            raise EmbeddingProviderError(
                "Embedding dimension does not match configured dimension "
                f"{self.config.dimension}"
            )
        if any(not isinstance(value, (int, float)) for value in vector):
            raise EmbeddingProviderError("Embedding provider returned non-numeric vector values")
