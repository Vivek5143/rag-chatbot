from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from src.config import Settings
from src.generation import (
    GenerationInputError,
    GenerationOutputError,
    GenerationProviderError,
    GenerationResult,
    OpenAIGenerationAdapter,
    build_generation_prompt,
    generate_answer,
)
from src.graph import REFUSAL_MESSAGE, build_graph
from src.grading import EvidenceGraderStructuredOutput
from src.models import RetrievedCandidate
from src.validation import GroundingValidationResult


def _candidate(chunk_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=f"evidence content for {chunk_id}",
        source="Ebook-Agentic-AI.pdf",
        page=3,
        source_hash="b" * 64,
        similarity_score=0.9,
        metadata={"chunk_id": chunk_id},
    )


class FakeRetrievalService:
    def __init__(self, candidates: list[RetrievedCandidate]) -> None:
        self.candidates = candidates

    def retrieve(self, *, question: str) -> list[RetrievedCandidate]:
        del question
        return self.candidates


class FakeGraderAdapter:
    def __init__(self, output: EvidenceGraderStructuredOutput) -> None:
        self.output = output

    def grade(self, *, question: str, candidates: list[RetrievedCandidate]) -> Any:
        del question, candidates
        return self.output


class FakeGenerationAdapter:
    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output = output if output is not None else {"answer": "Evidence-only answer."}
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate(
        self, *, question: str, supporting_candidates: list[RetrievedCandidate]
    ) -> Any:
        self.calls.append(
            {"question": question, "supporting_candidates": supporting_candidates}
        )
        if self.error:
            raise self.error
        return self.output


class FakeValidatorAdapter:
    def validate(
        self,
        *,
        question: str,
        answer: str,
        supporting_candidates: list[RetrievedCandidate],
    ) -> GroundingValidationResult:
        del question, answer, supporting_candidates
        return GroundingValidationResult(
            grounded=True,
            supported=True,
            unsupported_claims=[],
            missing_evidence=[],
            ambiguous_claims=[],
            entailment_score=0.9,
        )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        min_evidence_support_score=0.8,
        min_evidence_entailment_score=0.8,
    )


def _assessment(*, sufficient: bool, supporting_ids: list[str]) -> EvidenceGraderStructuredOutput:
    return EvidenceGraderStructuredOutput(
        relevant=sufficient,
        complete=sufficient,
        support_score=0.9,
        entailment_score=0.85,
        evidence_sufficient=sufficient,
        reason="supported" if sufficient else "insufficient",
        supporting_chunk_ids=supporting_ids,
        missing_aspects=[] if sufficient else ["missing information"],
    )


def _graph(
    *,
    candidates: list[RetrievedCandidate],
    assessment: EvidenceGraderStructuredOutput,
    generator: FakeGenerationAdapter,
):
    return build_graph(
        retrieval_service=FakeRetrievalService(candidates),
        grader_adapter=FakeGraderAdapter(assessment),
        generation_adapter=generator,
        validator_adapter=FakeValidatorAdapter(),
        settings=_settings(),
    )


def test_generate_answer_passes_question_and_only_supporting_candidates() -> None:
    supporting = [_candidate("support-1")]
    adapter = FakeGenerationAdapter()

    result = generate_answer(
        question=" What does the eBook establish? ",
        supporting_candidates=supporting,
        adapter=adapter,
    )

    assert result == GenerationResult(answer="Evidence-only answer.")
    assert adapter.calls == [
        {
            "question": "What does the eBook establish?",
            "supporting_candidates": supporting,
        }
    ]


def test_graph_excludes_irrelevant_retrieval_from_generation_context() -> None:
    supporting = _candidate("support-1")
    irrelevant = _candidate("irrelevant-2")
    adapter = FakeGenerationAdapter({"answer": "The supported answer."})

    result = _graph(
        candidates=[supporting, irrelevant],
        assessment=_assessment(sufficient=True, supporting_ids=["support-1"]),
        generator=adapter,
    ).invoke({"question": "What does the eBook establish?"})

    assert adapter.calls == [
        {
            "question": "What does the eBook establish?",
            "supporting_candidates": [supporting],
        }
    ]
    assert result["answer"] == "The supported answer."
    assert result["grounded"] is True
    assert result["refused"] is False
    assert result["confidence_score"] == 0.85


def test_generation_provider_failure_propagates_from_graph() -> None:
    with pytest.raises(GenerationProviderError):
        _graph(
            candidates=[_candidate("support-1")],
            assessment=_assessment(sufficient=True, supporting_ids=["support-1"]),
            generator=FakeGenerationAdapter(error=RuntimeError("unavailable")),
        ).invoke({"question": "Q"})


def test_openai_adapter_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

    fake_openai = ModuleType("openai")
    fake_openai.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    adapter = OpenAIGenerationAdapter.from_settings(
        settings=Settings(
            _env_file=None,
            openai_api_key="test-key",
            openai_model="configured-model",
        )
    )

    assert adapter._model == "configured-model"
    assert captured == {"api_key": "test-key"}


def test_empty_supporting_evidence_is_rejected_before_generator_call() -> None:
    adapter = FakeGenerationAdapter()

    with pytest.raises(GenerationInputError):
        generate_answer(question="Q", supporting_candidates=[], adapter=adapter)

    assert adapter.calls == []


def test_insufficient_evidence_refuses_without_calling_generator() -> None:
    adapter = FakeGenerationAdapter()
    result = _graph(
        candidates=[_candidate("retrieved-1")],
        assessment=_assessment(sufficient=False, supporting_ids=[]),
        generator=adapter,
    ).invoke({"question": "Q"})

    assert adapter.calls == []
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["refused"] is True


def test_prompt_enforces_evidence_only_with_delimited_provenance() -> None:
    prompt = build_generation_prompt(
        question="What does the evidence say?",
        supporting_candidates=[_candidate("support-1")],
    )

    normalized = prompt.lower()
    assert "only the supplied evidence" in normalized
    assert "outside knowledge" in normalized
    assert "do not infer" in normalized
    assert "<supporting_evidence>" in prompt
    assert "[chunk_id=support-1 | source=Ebook-Agentic-AI.pdf | page=3]" in prompt
    assert "evidence content for support-1" in prompt


def test_malformed_generation_output_is_an_operational_error() -> None:
    with pytest.raises(GenerationOutputError):
        generate_answer(
            question="Q",
            supporting_candidates=[_candidate("support-1")],
            adapter=FakeGenerationAdapter({"answer": "   "}),
        )
