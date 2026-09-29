from __future__ import annotations

from typing import Any

import pytest

from src.config import Settings
from src.models import EvidenceAssessment, RetrievedCandidate
from src.grading import (
    EvidenceGraderAdapter,
    EvidenceGraderOutputError,
    EvidenceGraderProviderError,
    EvidenceGraderStructuredOutput,
    grade_evidence,
)


def _candidate(chunk_id: str = "c1") -> RetrievedCandidate:
    return RetrievedCandidate(
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        source="Ebook-Agentic-AI.pdf",
        page=1,
        source_hash="a" * 64,
        similarity_score=0.5,
        metadata={
            "chunk_id": chunk_id,
            "content": f"content for {chunk_id}",
            "source": "Ebook-Agentic-AI.pdf",
            "page": 1,
            "source_hash": "a" * 64,
        },
    )


class FakeAdapter:
    def __init__(self, *, output: object, fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def grade(
        self, *, question: str, candidates: list[RetrievedCandidate]
    ) -> Any:
        self.calls.append({"question": question, "candidates": candidates})
        if self.fail:
            raise TimeoutError("provider timed out")
        return self.output


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        min_evidence_support_score=0.8,
        min_evidence_entailment_score=0.8,
    )


def test_grade_returns_sufficient_when_thresholds_and_flags_pass() -> None:
    candidates = [_candidate("b"), _candidate("c")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.9,
        entailment_score=0.85,
        evidence_sufficient=True,
        reason="supported",
        supporting_chunk_ids=["b"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    assessment: EvidenceAssessment = grade_evidence(
        question="Q",
        candidates=candidates,
        adapter=adapter,
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is True
    assert assessment.relevant is True
    assert assessment.complete is True
    assert assessment.supporting_chunk_ids == ["b"]


def test_grade_downgrades_when_support_score_below_threshold() -> None:
    candidates = [_candidate("b")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.1,
        entailment_score=0.9,
        evidence_sufficient=True,
        reason="low support",
        supporting_chunk_ids=["b"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    assessment = grade_evidence(
        question="Q",
        candidates=candidates,
        adapter=adapter,
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is False


def test_grade_returns_insufficient_when_relevant_is_false() -> None:
    candidates = [_candidate("b")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=False,
        complete=True,
        support_score=0.9,
        entailment_score=0.9,
        evidence_sufficient=False,
        reason="irrelevant",
        supporting_chunk_ids=[],
        missing_aspects=["not relevant"],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    assessment = grade_evidence(
        question="Q",
        candidates=candidates,
        adapter=adapter,
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is False


def test_grade_uses_multiple_supporting_chunk_ids_when_provided() -> None:
    candidates = [_candidate("b"), _candidate("c")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.95,
        entailment_score=0.95,
        evidence_sufficient=True,
        reason="supported",
        supporting_chunk_ids=["b", "c"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    assessment = grade_evidence(
        question="Q",
        candidates=candidates,
        adapter=adapter,
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is True
    assert set(assessment.supporting_chunk_ids) == {"b", "c"}


def test_grade_raises_operational_error_on_invalid_supporting_chunk_ids() -> None:
    candidates = [_candidate("b")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.9,
        entailment_score=0.9,
        evidence_sufficient=True,
        reason="supported",
        supporting_chunk_ids=["not-in-candidates"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    with pytest.raises(EvidenceGraderOutputError, match="supporting_chunk_ids"):
        grade_evidence(
            question="Q",
            candidates=candidates,
            adapter=adapter,
            settings=_settings(),
        )


def test_grade_raises_operational_error_on_invalid_structured_output_missing_field() -> None:
    candidates = [_candidate("b")]

    # missing supporting_chunk_ids
    adapter_output = {
        "relevant": True,
        "complete": True,
        "support_score": 0.9,
        "entailment_score": 0.9,
        "evidence_sufficient": True,
        "reason": "supported",
        "missing_aspects": [],
    }
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    with pytest.raises(EvidenceGraderOutputError):
        grade_evidence(
            question="Q",
            candidates=candidates,
            adapter=adapter,
            settings=_settings(),
        )


def test_grade_raises_operational_error_on_score_outside_valid_range() -> None:
    candidates = [_candidate("b")]

    adapter_output = {
        "relevant": True,
        "complete": True,
        "support_score": 1.5,
        "entailment_score": 0.9,
        "evidence_sufficient": True,
        "reason": "supported",
        "supporting_chunk_ids": ["b"],
        "missing_aspects": [],
    }
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    with pytest.raises(EvidenceGraderOutputError, match="support_score"):
        grade_evidence(
            question="Q",
            candidates=candidates,
            adapter=adapter,
            settings=_settings(),
        )


def test_grade_raises_operational_error_on_provider_failure() -> None:
    candidates = [_candidate("b")]
    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.9,
        entailment_score=0.9,
        evidence_sufficient=True,
        reason="supported",
        supporting_chunk_ids=["b"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output, fail=True)

    with pytest.raises(EvidenceGraderProviderError):
        grade_evidence(
            question="Q",
            candidates=candidates,
            adapter=adapter,
            settings=_settings(),
        )


def test_grade_empty_candidates_is_deterministic_and_does_not_call_adapter() -> None:
    class AdapterThatMustNotBeCalled:
        def __init__(self) -> None:
            self.called = False

        def grade(
            self, *, question: str, candidates: list[RetrievedCandidate]
        ) -> object:
            self.called = True
            raise AssertionError("adapter should not be called")

    adapter = AdapterThatMustNotBeCalled()

    assessment = grade_evidence(
        question="Q",
        candidates=[],
        adapter=adapter,  # type: ignore[arg-type]
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is False
    assert assessment.supporting_chunk_ids == []
    assert assessment.relevant is False
    assert assessment.complete is False


def test_retrieved_nonempty_does_not_imply_sufficient() -> None:
    candidates = [_candidate("b")]

    adapter_output = EvidenceGraderStructuredOutput(
        relevant=True,
        complete=True,
        support_score=0.9,
        entailment_score=0.9,
        evidence_sufficient=False,
        reason="agent says insufficient",
        supporting_chunk_ids=["b"],
        missing_aspects=[],
    )
    adapter: EvidenceGraderAdapter = FakeAdapter(output=adapter_output)

    assessment = grade_evidence(
        question="Q",
        candidates=candidates,
        adapter=adapter,
        settings=_settings(),
    )

    assert assessment.evidence_sufficient is False
