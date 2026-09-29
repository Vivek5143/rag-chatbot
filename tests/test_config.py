import pytest
from pydantic import ValidationError

from src.config import ConfigurationError, Settings


def test_settings_reports_missing_runtime_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_NAMESPACE"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings(_env_file=None)

    assert settings.missing_runtime_configuration() == (
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_NAMESPACE",
    )
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        settings.require_runtime_configuration()


def test_settings_accepts_dummy_runtime_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_NAMESPACE", "test-namespace")

    settings = Settings(_env_file=None)

    settings.require_runtime_configuration()
    assert settings.top_k == 4
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200
    assert settings.embedding_dimension == 1536
    assert settings.embedding_batch_size == 64
    assert settings.pinecone_upsert_batch_size == 100


def test_settings_reject_invalid_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

    with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
        Settings(_env_file=None)
