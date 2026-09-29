from __future__ import annotations

from typing import Any

import pytest

from src.config import Settings
from src.generation import GenerationResult
from src.grading import (
    EvidenceGraderOutputError,
    EvidenceGraderProviderError,
    EvidenceGraderStructuredOutput,
)
from src.graph import REFUSAL_MESSAGE, build_graph
from src.models import RetrievedCandidate
from src.retrieval import RetrievalProviderError
from src.validation import GroundingValidationProviderError, GroundingValidationResult


def _candidate(chunk_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        source="Ebook-Agentic-AI.pdf",
        page=1,
        source_hash="a" * 64,
        similarity_score=0.8,
        metadata={"chunk_id": chunk_id},
    )


class FakeRetrievalService:
    def __init__(self, candidates: list[RetrievedCandidate], error: Exception | None = None) -> None:
        self.candidates = candidates
        self.error = error
        self.questions: list[str] = []

    def retrieve(self, *, question: str) -> list[RetrievedCandidate]:
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.candidates


class FakeGraderAdapter:
    def __init__(self, output: Any = None, error: Exception | None = None) -> None:
        self.output = output
        self.error = error
        self.calls: list[dict[str, object]] = []

    def grade(self, *, question: str, candidates: list[RetrievedCandidate]) -> Any:
        self.calls.append({"question": question, "candidates": candidates})
        if self.error:
            raise self.error
        return self.output


class FakeGenerationAdapter:
    def __init__(self, answer: str = "Generated from supporting evidence.") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def generate(
        self, *, question: str, supporting_candidates: list[RetrievedCandidate]
    ) -> GenerationResult:
        self.calls.append(
            {"question": question, "supporting_candidates": supporting_candidates}
        )
        return GenerationResult(answer=self.answer)


class FakeValidatorAdapter:
    def __init__(
        self,
        result: GroundingValidationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else GroundingValidationResult(
            grounded=True,
            supported=True,
            unsupported_claims=[],
            missing_evidence=[],
            ambiguous_claims=[],
            entailment_score=0.9,
        )
        self.error = error
        self.calls: list[dict[str, object]] = []

    def validate(
        self,
        *,
        question: str,
        answer: str,
        supporting_candidates: list[RetrievedCandidate],
    ) -> GroundingValidationResult:
        self.calls.append(
            {
                "question": question,
                "answer": answer,
                "supporting_candidates": supporting_candidates,
            }
        )
        if self.error:
            raise self.error
        return self.result


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        min_evidence_support_score=0.8,
        min_evidence_entailment_score=0.8,
    )


def _output(*, sufficient: bool, supporting_ids: list[str]) -> EvidenceGraderStructuredOutput:
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
    retrieval: FakeRetrievalService,
    grader: FakeGraderAdapter,
    generator: FakeGenerationAdapter | None = None,
    validator: FakeValidatorAdapter | None = None,
):
    return build_graph(
        retrieval_service=retrieval,
        grader_adapter=grader,
        generation_adapter=generator or FakeGenerationAdapter(),
        validator_adapter=validator or FakeValidatorAdapter(),
        settings=_settings(),
    )


def test_question_reaches_retrieval_and_candidates_reach_grading() -> None:
    candidates = [_candidate("c1"), _candidate("c2")]
    retrieval = FakeRetrievalService(candidates)
    grader = FakeGraderAdapter(_output(sufficient=False, supporting_ids=[]))

    _graph(retrieval, grader).invoke({"question": "What is Agentic AI?"})

    assert retrieval.questions == ["What is Agentic AI?"]
    assert grader.calls[0] == {"question": "What is Agentic AI?", "candidates": candidates}


def test_sufficient_evidence_uses_generation_then_validation_path() -> None:
    candidate = _candidate("c1")
    generator = FakeGenerationAdapter()
    validator = FakeValidatorAdapter()
    result = _graph(
        FakeRetrievalService([candidate]),
        FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c1"])),
        generator=generator,
        validator=validator,
    ).invoke({"question": "What is Agentic AI?"})

    assert result["supporting_candidates"] == [candidate]
    assert result["answer"] == "Generated from supporting evidence."
    assert result["refused"] is False
    assert result["grounded"] is True
    assert generator.calls == [
        {
            "question": "What is Agentic AI?",
            "supporting_candidates": [candidate],
        }
    ]
    assert validator.calls == [
        {
            "question": "What is Agentic AI?",
            "answer": "Generated from supporting evidence.",
            "supporting_candidates": [candidate],
        }
    ]


def test_insufficient_evidence_follows_refusal_with_exact_state() -> None:
    result = _graph(
        FakeRetrievalService([_candidate("c1")]),
        FakeGraderAdapter(_output(sufficient=False, supporting_ids=[])),
    ).invoke({"question": "What is unrelated?"})

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["confidence_score"] == 0.0
    assert result["grounded"] is False
    assert result["refused"] is True


def test_refusal_preserves_retrieved_context() -> None:
    candidates = [_candidate("c1"), _candidate("c2")]
    result = _graph(
        FakeRetrievalService(candidates),
        FakeGraderAdapter(_output(sufficient=False, supporting_ids=[])),
    ).invoke({"question": "What is unrelated?"})

    assert result["retrieved_candidates"] == candidates


def test_supporting_candidates_contain_only_grader_selected_chunks() -> None:
    candidates = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    result = _graph(
        FakeRetrievalService(candidates),
        FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c3", "c1"])),
    ).invoke({"question": "Q"})

    assert [candidate.chunk_id for candidate in result["supporting_candidates"]] == ["c3", "c1"]


def test_invalid_grader_output_propagates_as_operational_error() -> None:
    with pytest.raises(EvidenceGraderOutputError):
        _graph(
            FakeRetrievalService([_candidate("c1")]),
            FakeGraderAdapter({"relevant": True}),
        ).invoke({"question": "Q"})


def test_invalid_supporting_chunk_id_propagates_as_operational_error() -> None:
    with pytest.raises(EvidenceGraderOutputError):
        _graph(
            FakeRetrievalService([_candidate("c1")]),
            FakeGraderAdapter(_output(sufficient=True, supporting_ids=["unknown"])),
        ).invoke({"question": "Q"})


def test_retrieval_failure_propagates() -> None:
    with pytest.raises(RetrievalProviderError):
        _graph(
            FakeRetrievalService([], error=RetrievalProviderError("pinecone query failed")),
            FakeGraderAdapter(),
        ).invoke({"question": "Q"})


def test_grader_provider_failure_propagates() -> None:
    with pytest.raises(EvidenceGraderProviderError):
        _graph(
            FakeRetrievalService([_candidate("c1")]),
            FakeGraderAdapter(error=EvidenceGraderProviderError("provider failed")),
        ).invoke({"question": "Q"})


def test_empty_retrieval_follows_normal_refusal_path() -> None:
    grader = FakeGraderAdapter()
    result = _graph(FakeRetrievalService([]), grader).invoke({"question": "Q"})

    assert result["retrieved_candidates"] == []
    assert result["answer"] == REFUSAL_MESSAGE
    assert result["refused"] is True
    assert grader.calls == []


def test_sufficient_evidence_is_grounded_only_after_validation() -> None:
    result = _graph(
        FakeRetrievalService([_candidate("c1")]),
        FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c1"])),
    ).invoke({"question": "Q"})

    assert result["answer"] == "Generated from supporting evidence."
    assert result["grounded"] is True


def test_validator_rejection_replaces_generated_answer_with_refusal() -> None:
    candidates = [_candidate("c1"), _candidate("c2")]
    validator = FakeValidatorAdapter(
        GroundingValidationResult(
            grounded=False,
            supported=False,
            unsupported_claims=["Unsupported generated claim."],
            missing_evidence=[],
            ambiguous_claims=[],
            entailment_score=0.2,
        )
    )
    result = _graph(
        FakeRetrievalService(candidates),
        FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c1"])),
        validator=validator,
    ).invoke({"question": "Q"})

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["answer"] != "Generated from supporting evidence."
    assert result["grounded"] is False
    assert result["refused"] is True
    assert result["confidence_score"] == 0.0
    assert result["retrieved_candidates"] == candidates


def test_validator_uncertainty_follows_refusal_path() -> None:
    validator = FakeValidatorAdapter(
        GroundingValidationResult(
            grounded=False,
            supported=False,
            unsupported_claims=[],
            missing_evidence=[],
            ambiguous_claims=["Claim is ambiguous."],
            entailment_score=0.5,
        )
    )
    result = _graph(
        FakeRetrievalService([_candidate("c1")]),
        FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c1"])),
        validator=validator,
    ).invoke({"question": "Q"})

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["grounded"] is False


def test_validator_provider_failure_propagates_as_operational_error() -> None:
    with pytest.raises(GroundingValidationProviderError):
        _graph(
            FakeRetrievalService([_candidate("c1")]),
            FakeGraderAdapter(_output(sufficient=True, supporting_ids=["c1"])),
            validator=FakeValidatorAdapter(
                error=GroundingValidationProviderError("validator unavailable")
            ),
        ).invoke({"question": "Q"})
