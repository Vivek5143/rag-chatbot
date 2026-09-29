from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from src.config import Settings
from src.models import RetrievedCandidate
from src.validation import (
    GroundingValidationOutputError,
    GroundingValidationProviderError,
    OpenAIGroundingValidatorAdapter,
    build_validation_prompt,
    validate_grounding,
)
import src.validation as validation_module


def _candidate(chunk_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=f"evidence for {chunk_id}",
        source="Ebook-Agentic-AI.pdf",
        page=4,
        source_hash="c" * 64,
        similarity_score=0.91,
        metadata={"chunk_id": chunk_id},
    )


def _settings(*, threshold: float = 0.8) -> Settings:
    return Settings(_env_file=None, min_evidence_entailment_score=threshold)


def _output(
    *,
    grounded: bool = True,
    supported: bool = True,
    unsupported_claims: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    ambiguous_claims: list[str] | None = None,
    entailment_score: float = 0.9,
) -> dict[str, object]:
    return {
        "grounded": grounded,
        "supported": supported,
        "unsupported_claims": unsupported_claims or [],
        "missing_evidence": missing_evidence or [],
        "ambiguous_claims": ambiguous_claims or [],
        "entailment_score": entailment_score,
    }


class FakeValidatorAdapter:
    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output = output if output is not None else _output()
        self.error = error
        self.calls: list[dict[str, object]] = []

    def validate(
        self,
        *,
        question: str,
        answer: str,
        supporting_candidates: list[RetrievedCandidate],
    ) -> Any:
        self.calls.append(
            {
                "question": question,
                "answer": answer,
                "supporting_candidates": supporting_candidates,
            }
        )
        if self.error:
            raise self.error
        return self.output


def _validate(
    *,
    answer: str = "A supported answer.",
    candidates: list[RetrievedCandidate] | None = None,
    adapter: FakeValidatorAdapter | None = None,
    threshold: float = 0.8,
):
    return validate_grounding(
        question="What does the eBook establish?",
        answer=answer,
        supporting_candidates=candidates if candidates is not None else [_candidate("c1")],
        adapter=adapter or FakeValidatorAdapter(),
        settings=_settings(threshold=threshold),
    )


def test_fully_supported_answer_is_grounded() -> None:
    result = _validate()

    assert result.grounded is True
    assert result.supported is True
    assert result.unsupported_claims == []


def test_partially_supported_answer_is_rejected() -> None:
    result = _validate(
        adapter=FakeValidatorAdapter(
            _output(supported=False, missing_evidence=["the answer's second claim"])
        )
    )

    assert result.grounded is False
    assert result.missing_evidence == ["the answer's second claim"]


def test_unsupported_claim_is_rejected() -> None:
    result = _validate(
        adapter=FakeValidatorAdapter(
            _output(
                grounded=False,
                supported=False,
                unsupported_claims=["Agentic AI was invented in 2018."],
            )
        )
    )

    assert result.grounded is False
    assert result.unsupported_claims == ["Agentic AI was invented in 2018."]


def test_multiple_supported_claims_can_be_grounded() -> None:
    result = _validate(
        answer="The system uses planning. It also uses memory.",
        adapter=FakeValidatorAdapter(_output()),
    )

    assert result.grounded is True


def test_ambiguous_answer_is_rejected() -> None:
    result = _validate(
        adapter=FakeValidatorAdapter(
            _output(
                grounded=False,
                supported=False,
                ambiguous_claims=["The scope of autonomous behavior is ambiguous."],
            )
        )
    )

    assert result.grounded is False
    assert result.ambiguous_claims


def test_empty_supporting_evidence_fails_closed_without_provider_call() -> None:
    adapter = FakeValidatorAdapter()
    result = _validate(candidates=[], adapter=adapter)

    assert result.grounded is False
    assert result.missing_evidence == ["no supporting evidence was available for validation"]
    assert adapter.calls == []


def test_blank_unsupported_claim_description_is_invalid_output() -> None:
    with pytest.raises(GroundingValidationOutputError):
        _validate(adapter=FakeValidatorAdapter(_output(unsupported_claims=[" "])))


def test_entailment_threshold_is_enforced() -> None:
    result = _validate(
        adapter=FakeValidatorAdapter(_output(entailment_score=0.79)), threshold=0.8
    )

    assert result.grounded is False


def test_invalid_entailment_score_is_an_operational_error() -> None:
    with pytest.raises(GroundingValidationOutputError):
        _validate(adapter=FakeValidatorAdapter(_output(entailment_score=1.1)))


def test_malformed_structured_output_is_an_operational_error() -> None:
    with pytest.raises(GroundingValidationOutputError):
        _validate(adapter=FakeValidatorAdapter({"grounded": True}))


def test_provider_failure_is_an_operational_error() -> None:
    with pytest.raises(GroundingValidationProviderError):
        _validate(adapter=FakeValidatorAdapter(error=TimeoutError("timed out")))


def test_validator_receives_only_the_supplied_supporting_candidates() -> None:
    supporting = [_candidate("support-1")]
    adapter = FakeValidatorAdapter()

    _validate(candidates=supporting, adapter=adapter)

    assert adapter.calls == [
        {
            "question": "What does the eBook establish?",
            "answer": "A supported answer.",
            "supporting_candidates": supporting,
        }
    ]


def test_validator_has_no_retrieval_dependency() -> None:
    assert not hasattr(validation_module, "RetrievalService")


def test_validation_prompt_uses_only_evidence_and_prohibits_outside_knowledge() -> None:
    prompt = build_validation_prompt(
        question="What does the evidence establish?",
        answer="A concise answer.",
        supporting_candidates=[_candidate("support-1")],
    )

    normalized = prompt.lower()
    assert "only source of truth" in normalized
    assert "do not use outside knowledge" in normalized
    assert "do not infer missing facts" in normalized
    assert "<answer_to_validate>" in prompt
    assert "<supporting_evidence>" in prompt
    assert "[chunk_id=support-1 | source=Ebook-Agentic-AI.pdf | page=4]" in prompt


def test_openai_validator_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    adapter = OpenAIGroundingValidatorAdapter.from_settings(
        settings=Settings(
            _env_file=None,
            openai_api_key="test-key",
            openai_model="configured-model",
        )
    )

    assert adapter._model == "configured-model"
    assert captured == {"api_key": "test-key"}
