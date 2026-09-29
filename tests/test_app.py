import pytest
from fastapi.testclient import TestClient

from app import app
client = TestClient(app)


def test_health_endpoint_reports_api_liveness() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": "api"}


def test_default_chat_returns_safe_503_when_runtime_configuration_is_missing() -> None:
    response = client.post("/chat", json={"query": "What is Agentic AI?"})

    assert response.status_code == 503
    assert response.json() == {"detail": "RAG service is currently unavailable."}


def test_chat_rejects_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("PINECONE_API_KEY", "pk-test")
    monkeypatch.setenv("PINECONE_NAMESPACE", "agentic-ai-test")
    from src.config import get_settings
    get_settings.cache_clear()

    response = client.post("/chat", json={"query": "   "})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "query"]
