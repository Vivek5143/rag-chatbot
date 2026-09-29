from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from src.config import Settings, get_settings
from src.models import EvidenceAssessment, RetrievedCandidate


class EvidenceGraderError(RuntimeError):
    """Base class for evidence grading failures."""


class EvidenceGraderProviderError(EvidenceGraderError):
    """Raised when the evidence grader provider fails (timeouts, transport, etc.)."""


class EvidenceGraderOutputError(EvidenceGraderError):
    """Raised when the grader returns malformed/invalid structured output."""


class EvidenceGraderStructuredOutput(BaseModel):
    """Adapter output contract for structured evidence grading."""

    relevant: bool
    complete: bool

    support_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(ge=0.0, le=1.0)

    evidence_sufficient: bool
    reason: str = Field(..., min_length=1)

    supporting_chunk_ids: list[str]
    missing_aspects: list[str]


class EvidenceGraderAdapter(Protocol):
    """Mockable adapter interface for evidence grading."""

    def grade(
        self, *, question: str, candidates: list[RetrievedCandidate]
    ) -> Any:
        """Return structured evidence grading output."""


def grade_evidence(
    *,
    question: str,
    candidates: list[RetrievedCandidate],
    adapter: EvidenceGraderAdapter,
    settings: Settings | None = None,
) -> EvidenceAssessment:
    active_settings = settings or get_settings()

    if not candidates:
        return EvidenceAssessment(
            relevant=False,
            complete=False,
            support_score=0.0,
            entailment_score=0.0,
            evidence_sufficient=False,
            reason="no_retrieval",
            supporting_chunk_ids=[],
            missing_aspects=["no documents were retrieved"],
        )

    try:
        raw = adapter.grade(question=question, candidates=candidates)
    except EvidenceGraderError:
        raise
    except Exception as error:
        raise EvidenceGraderProviderError("evidence grader provider failed") from error

    try:
        parsed = (
            raw
            if isinstance(raw, EvidenceGraderStructuredOutput)
            else EvidenceGraderStructuredOutput.model_validate(raw)
        )
    except ValidationError as error:
        invalid_fields: list[str] = []
        try:
            invalid_fields = [
                str(err.get("loc")[0])
                for err in error.errors()
                if err.get("loc")
            ]
        except Exception:
            invalid_fields = []

        suffix = (
            ": invalid " + ", ".join(sorted(set(invalid_fields)))
            if invalid_fields
            else ""
        )
        raise EvidenceGraderOutputError("grader output is malformed" + suffix) from error

    candidate_ids = {candidate.chunk_id for candidate in candidates}
    invalid_ids = [
        chunk_id for chunk_id in parsed.supporting_chunk_ids if chunk_id not in candidate_ids
    ]
    if invalid_ids:
        raise EvidenceGraderOutputError(
            "supporting_chunk_ids must exist in retrieved candidates: "
            + ", ".join(invalid_ids)
        )

    threshold_support_ok = parsed.support_score >= active_settings.min_evidence_support_score
    threshold_entail_ok = (
        parsed.entailment_score >= active_settings.min_evidence_entailment_score
    )

    effective = (
        parsed.evidence_sufficient
        and parsed.relevant
        and parsed.complete
        and threshold_support_ok
        and threshold_entail_ok
        and (len(parsed.missing_aspects) == 0)
        and (len(parsed.supporting_chunk_ids) > 0)
    )

    # EvidenceAssessment's schema validation enforces its own internal invariants.
    try:
        return EvidenceAssessment(
            relevant=parsed.relevant,
            complete=parsed.complete,
            support_score=float(parsed.support_score),
            entailment_score=float(parsed.entailment_score),
            evidence_sufficient=effective,
            reason=parsed.reason,
            supporting_chunk_ids=list(parsed.supporting_chunk_ids),
            missing_aspects=list(parsed.missing_aspects),
        )
    except ValidationError as error:
        raise EvidenceGraderOutputError("evidence assessment is invalid") from error


class OpenAIEvidenceGraderAdapter:
    """OpenAI-backed evidence grader adapter.

    Note: unit tests should inject a fake adapter rather than calling this.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 20.0,
        base_url: str = "",
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must not be blank")
        self._model = model
        self._timeout_seconds = timeout_seconds

        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover
            raise EvidenceGraderProviderError("openai package is required") from error

        # ``base_url`` is empty by default so the client targets the official
        # OpenAI endpoint; an explicit value redirects it at an OpenAI-compatible
        # provider.
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": timeout_seconds,
        }
        if base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        self._client = OpenAI(**client_kwargs)

    def grade(
        self, *, question: str, candidates: list[RetrievedCandidate]
    ) -> EvidenceGraderStructuredOutput:
        # Lazy import / construction: already safe since OpenAIEvidenceGraderAdapter
        # should be instantiated only in runtime configuration.
        import json

        # Keep prompt concise: evidence grading is structured and bounded.
        evidence_blocks = []
        for candidate in candidates:
            evidence_blocks.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "source": candidate.source,
                    "page": candidate.page,
                    "content": candidate.content,
                }
            )

        prompt = {
            "question": question,
            "candidates": evidence_blocks,
            "instructions": "Return STRICT JSON for evidence grading. Do not generate an answer."
            " Determine relevant/complete/support_score/entailment_score/evidence_sufficient."
            " supporting_chunk_ids must list only chunk_ids from candidates that support the answer."
            " missing_aspects must list any required aspects that are missing.",
            "output_schema": EvidenceGraderStructuredOutput.model_json_schema(),
        }

        try:
            parsed_response = self._client.responses.parse(
                model=self._model,
                input=json.dumps(prompt),
                text_format=EvidenceGraderStructuredOutput,
            )
        except Exception as error:  # pragma: no cover
            raise EvidenceGraderProviderError("evidence grader provider failed") from error

        parsed = parsed_response.output_parsed
        if parsed is None:  # pragma: no cover
            raise EvidenceGraderOutputError("grader output is malformed: missing parsed result")

        return parsed
