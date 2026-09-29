"""Local FastAPI/TestClient smoke test for Phase 8.

Covers the five mandated cases without contacting OpenAI or Pinecone:
  1. GET /health
  2. POST /chat with an invalid query -> 422
  3. POST /chat with a mocked grounded result -> 200
  4. POST /chat with a mocked refusal -> 200
  5. POST /chat with a mocked operational failure -> 503
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from src.api import SERVICE_UNAVAILABLE_DETAIL, create_app
from src.config import Settings
from src.graph import REFUSAL_MESSAGE
from src.models import RetrievedCandidate
from src.retrieval import RetrievalProviderError


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


class FakeGraph:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def invoke(self, input: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(input)
        if self.error:
            raise self.error
        return self.result


class FakeGraphFactory:
    def __init__(self, graph: FakeGraph) -> None:
        self.graph = graph
        self.calls = 0

    def __call__(self) -> FakeGraph:
        self.calls += 1
        return self.graph


def _client(graph: FakeGraph) -> tuple[TestClient, FakeGraphFactory]:
    factory = FakeGraphFactory(graph)
    app = create_app(
        graph_factory=factory,
        settings_factory=lambda: Settings(_env_file=None, top_k=4),
        context_limit=4,
    )
    return TestClient(app), factory


def main() -> None:
    failures: list[str] = []

    # 1. GET /health
    from app import app as default_app
    health_client = TestClient(default_app)
    r = health_client.get("/health")
    if r.status_code != 200 or r.json() != {"status": "ok", "phase": "api"}:
        failures.append(f"/health: {r.status_code} {r.text}")

    # 2. Invalid query -> 422
    graph = FakeGraph()
    client, _ = _client(graph)
    r = client.post("/chat", json={"query": "   "})
    if r.status_code != 422 or graph.calls:
        failures.append(f"invalid query: {r.status_code} calls={graph.calls}")

    # 3. Mocked grounded result -> 200
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    grounded = {
        "answer": "A validated answer.",
        "grounded": True,
        "refused": False,
        "confidence_score": 0.85,
        "retrieved_candidates": candidates,
    }
    client, factory = _client(FakeGraph(grounded))
    r = client.post("/chat", json={"query": "What is Agentic AI?"})
    body = r.json()
    if r.status_code != 200:
        failures.append(f"grounded: {r.status_code} {r.text}")
    elif body["final_answer"] != "A validated answer.":
        failures.append(f"grounded answer: {body}")
    elif body["grounded"] is not True or body["confidence_score"] != 0.85:
        failures.append(f"grounded flags: {body}")
    elif len(body["retrieved_context"]) != 3:
        failures.append(f"grounded context count: {len(body['retrieved_context'])}")
    elif body["retrieved_context"][0]["chunk_id"] != "c1":
        failures.append(f"grounded provenance: {body['retrieved_context'][0]}")
    elif factory.calls != 1:
        failures.append(f"grounded factory calls: {factory.calls}")

    # 4. Mocked refusal -> 200
    refusal = {
        "answer": REFUSAL_MESSAGE,
        "grounded": False,
        "refused": True,
        "confidence_score": 0.0,
        "retrieved_candidates": [_candidate("c1")],
    }
    client, _ = _client(FakeGraph(refusal))
    r = client.post("/chat", json={"query": "What is unsupported?"})
    body = r.json()
    if r.status_code != 200:
        failures.append(f"refusal: {r.status_code} {r.text}")
    elif body["final_answer"] != REFUSAL_MESSAGE:
        failures.append(f"refusal message: {body}")
    elif body["grounded"] is not False or body["confidence_score"] != 0.0:
        failures.append(f"refusal flags: {body}")
    elif body["retrieved_context"][0]["chunk_id"] != "c1":
        failures.append(f"refusal diagnostics: {body}")

    # 5. Mocked operational failure -> 503
    client, _ = _client(FakeGraph(error=RetrievalProviderError("pinecone unavailable")))
    r = client.post("/chat", json={"query": "Q"})
    if r.status_code != 503 or r.json() != {"detail": SERVICE_UNAVAILABLE_DETAIL}:
        failures.append(f"operational: {r.status_code} {r.text}")
    if "pinecone unavailable" in r.text:
        failures.append("operational error leaked internal detail")

    if failures:
        print("SMOKE TEST FAILED")
        for line in failures:
            print(" -", line)
        raise SystemExit(1)

    print("SMOKE TEST PASSED: /health, 422, grounded 200, refusal 200, 503")


if __name__ == "__main__":
    main()