from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api import SERVICE_UNAVAILABLE_DETAIL, create_app
from src.config import ConfigurationError, Settings
from src.generation import GenerationOutputError, GenerationProviderError
from src.graph import REFUSAL_MESSAGE
from src.grading import EvidenceGraderOutputError, EvidenceGraderProviderError
from src.models import RetrievedCandidate
from src.retrieval import RetrievalEmbeddingError, RetrievalProviderError
from src.validation import GroundingValidationOutputError, GroundingValidationProviderError


def _candidate(chunk_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        source="Ebook-Agentic-AI.pdf",
        page=2,
        source_hash="d" * 64,
        similarity_score=0.87,
        metadata={"chunk_id": chunk_id},
    )


def _grounded_state(*, candidates: list[RetrievedCandidate] | None = None) -> dict[str, Any]:
    return {
        "answer": "A validated answer.",
        "grounded": True,
        "refused": False,
        "confidence_score": 0.85,
        "retrieved_candidates": candidates if candidates is not None else [_candidate("c1")],
    }


def _refusal_state(
    *,
    candidates: list[RetrievedCandidate] | None = None,
    evidence_sufficient: bool = False,
    reason: str = "insufficient_evidence",
) -> dict[str, Any]:
    return {
        "answer": REFUSAL_MESSAGE,
        "grounded": False,
        "refused": True,
        "confidence_score": 0.0,
        "retrieved_candidates": candidates if candidates is not None else [_candidate("c1")],
        "evidence_assessment": SimpleNamespace(
            evidence_sufficient=evidence_sufficient,
            reason=reason,
        ),
    }


class FakeGraph:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result or _grounded_state()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(input)
        if self.error:
            raise self.error
        return self.result


class FakeGraphFactory:
    def __init__(self, graph: FakeGraph | None = None, error: Exception | None = None) -> None:
        self.graph = graph or FakeGraph()
        self.error = error
        self.calls = 0

    def __call__(self) -> FakeGraph:
        self.calls += 1
        if self.error:
            raise self.error
        return self.graph


def _client(
    graph: FakeGraph | None = None,
    *,
    factory_error: Exception | None = None,
    context_limit: int = 4,
) -> tuple[TestClient, FakeGraphFactory]:
    factory = FakeGraphFactory(graph=graph, error=factory_error)
    app = create_app(
        graph_factory=factory,
        settings_factory=lambda: Settings(_env_file=None, top_k=4),
        context_limit=context_limit,
    )
    return TestClient(app), factory


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {},
        {"query": 123},
        {"query": "x" * (2000 + 1)},
    ],
)
def test_chat_request_validation_returns_422(payload: dict[str, object]) -> None:
    graph = FakeGraph()
    client, _ = _client(graph)

    response = client.post("/chat", json=payload)

    assert response.status_code == 422
    assert graph.calls == []
    assert response.json()["detail"][0]["loc"] == ["body", "query"]


def test_grounded_graph_result_returns_provenance_rich_response() -> None:
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    graph = FakeGraph(_grounded_state(candidates=candidates))
    client, factory = _client(graph, context_limit=2)

    response = client.post("/chat", json={"query": "What is Agentic AI?"})

    assert response.status_code == 200
    assert response.json() == {
        "final_answer": "A validated answer.",
        "retrieved_context": [
            {
                "chunk_id": "c1",
                "source": "Ebook-Agentic-AI.pdf",
                "page": 2,
                "content": "content for c1",
                "source_hash": "d" * 64,
                "similarity_score": 0.87,
            },
            {
                "chunk_id": "c2",
                "source": "Ebook-Agentic-AI.pdf",
                "page": 2,
                "content": "content for c2",
                "source_hash": "d" * 64,
                "similarity_score": 0.87,
            },
        ],
        "confidence_score": 0.85,
        "grounded": True,
        "refusal_reason": None,
    }
    assert graph.calls == [{"question": "What is Agentic AI?"}]
    assert factory.calls == 1


def test_evidence_refusal_returns_200_and_preserves_diagnostics() -> None:
    graph = FakeGraph(_refusal_state(candidates=[_candidate("c1")]))
    client, _ = _client(graph)

    response = client.post("/chat", json={"query": "What is unsupported?"})

    body = response.json()
    assert response.status_code == 200
    assert body["final_answer"] == REFUSAL_MESSAGE
    assert body["confidence_score"] == 0.0
    assert body["grounded"] is False
    assert body["refusal_reason"] == "insufficient_evidence"
    assert body["retrieved_context"][0]["chunk_id"] == "c1"


def test_grounding_refusal_never_exposes_unsupported_generated_answer() -> None:
    graph = FakeGraph(
        _refusal_state(
            evidence_sufficient=True,
            reason="supported",
        )
    )
    client, _ = _client(graph)

    response = client.post("/chat", json={"query": "Q"})

    body = response.json()
    assert response.status_code == 200
    assert body["final_answer"] == REFUSAL_MESSAGE
    assert "unsupported generated answer" not in body["final_answer"]
    assert body["confidence_score"] == 0.0
    assert body["refusal_reason"] == "grounding_validation_failed"


@pytest.mark.parametrize(
    "error",
    [
        RetrievalProviderError("pinecone unavailable"),
        RetrievalEmbeddingError("embedding unavailable"),
        EvidenceGraderProviderError("grader unavailable"),
        EvidenceGraderOutputError("invalid grader output"),
        GenerationProviderError("generation unavailable"),
        GenerationOutputError("invalid generation output"),
        GroundingValidationProviderError("validator unavailable"),
        GroundingValidationOutputError("invalid validator output"),
    ],
)
def test_operational_graph_errors_return_safe_503(error: Exception) -> None:
    client, _ = _client(FakeGraph(error=error))

    response = client.post("/chat", json={"query": "Q"})

    assert response.status_code == 503
    assert response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}


def test_configuration_error_during_graph_construction_returns_503() -> None:
    client, factory = _client(factory_error=ConfigurationError("missing API key"))

    response = client.post("/chat", json={"query": "Q"})

    assert response.status_code == 503
    assert response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}
    assert factory.calls == 1


def test_unexpected_exception_returns_safe_503_without_internal_details() -> None:
    client, _ = _client(FakeGraph(error=RuntimeError("secret-token and traceback")))

    response = client.post("/chat", json={"query": "Q"})

    assert response.status_code == 503
    assert response.json() == {"detail": SERVICE_UNAVAILABLE_DETAIL}
    assert "secret-token" not in response.text
    assert "traceback" not in response.text


def test_graph_is_constructed_once_and_invoked_once_per_request() -> None:
    graph = FakeGraph()
    client, factory = _client(graph)

    first = client.post("/chat", json={"query": "First question"})
    second = client.post("/chat", json={"query": "Second question"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert factory.calls == 1
    assert graph.calls == [
        {"question": "First question"},
        {"question": "Second question"},
    ]
