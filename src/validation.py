"""Independent, fail-closed grounding validation for generated answers."""

from __future__ import annotations

import json
from math import isfinite
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from src.config import Settings, get_settings
from src.models import RetrievedCandidate


class GroundingValidationError(RuntimeError):
    """Base class for grounding-validation failures."""


class GroundingValidationInputError(GroundingValidationError, ValueError):
    """Raised before malformed validation input reaches a provider."""


class GroundingValidationProviderError(GroundingValidationError):
    """Raised when the grounding-validation provider cannot complete a request."""


class GroundingValidationOutputError(GroundingValidationError):
    """Raised when structured validation output is malformed or unusable."""


class GroundingValidatorStructuredOutput(BaseModel):
    """Structured, claim-level assessment returned by a validation adapter."""

    grounded: bool
    supported: bool
    unsupported_claims: list[str]
    missing_evidence: list[str]
    ambiguous_claims: list[str]
    entailment_score: float = Field(ge=0.0, le=1.0)

    @field_validator("unsupported_claims", "missing_evidence", "ambiguous_claims")
    @classmethod
    def require_non_blank_claim_descriptions(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("claim descriptions must not be blank")
        return normalized

    @field_validator("entailment_score")
    @classmethod
    def require_finite_entailment_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("entailment_score must be finite")
        return value


class GroundingValidationResult(GroundingValidatorStructuredOutput):
    """Effective fail-closed grounding decision used by the graph."""


class GroundingValidatorAdapter(Protocol):
    """Mockable provider port for independent claim-support validation."""

    def validate(
        self,
        *,
        question: str,
        answer: str,
        supporting_candidates: list[RetrievedCandidate],
    ) -> Any:
        """Assess the answer only against the supplied supporting evidence."""


def _normalize_non_blank(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise GroundingValidationInputError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise GroundingValidationInputError(f"{field_name} must not be blank")
    return normalized


def format_validation_evidence(candidates: list[RetrievedCandidate]) -> str:
    """Format only the generation evidence boundary with provenance labels."""

    if not candidates:
        raise GroundingValidationInputError("supporting candidates must not be empty")
    return "\n\n".join(
        "[chunk_id={chunk_id} | source={source} | page={page}]\n{content}".format(
            chunk_id=candidate.chunk_id,
            source=candidate.source,
            page=candidate.page,
            content=candidate.content,
        )
        for candidate in candidates
    )


def build_validation_prompt(
    *, question: str, answer: str, supporting_candidates: list[RetrievedCandidate]
) -> str:
    """Build the validation-only prompt; it neither rewrites nor expands evidence."""

    normalized_question = _normalize_non_blank(question, field_name="question")
    normalized_answer = _normalize_non_blank(answer, field_name="answer")
    evidence = format_validation_evidence(supporting_candidates)
    return f"""You are an independent grounding validator. The supplied evidence is
the only source of truth. Do not use outside knowledge. Evaluate every material
claim in the answer against the supplied evidence. Do not infer missing facts.
Mark uncertain, ambiguous, or unsupported claims as unsupported or ambiguous.
Do not rewrite, improve, or answer the question; validate only. Return structured
output only with grounded, supported, unsupported_claims, missing_evidence,
ambiguous_claims, and entailment_score.

<question>
{normalized_question}
</question>

<answer_to_validate>
{normalized_answer}
</answer_to_validate>

<supporting_evidence>
{evidence}
</supporting_evidence>
"""


def _no_evidence_result() -> GroundingValidationResult:
    return GroundingValidationResult(
        grounded=False,
        supported=False,
        unsupported_claims=[],
        missing_evidence=["no supporting evidence was available for validation"],
        ambiguous_claims=[],
        entailment_score=0.0,
    )


def validate_grounding(
    *,
    question: str,
    answer: str,
    supporting_candidates: list[RetrievedCandidate],
    adapter: GroundingValidatorAdapter,
    settings: Settings | None = None,
) -> GroundingValidationResult:
    """Return a fail-closed decision without retrieving or expanding evidence."""

    normalized_question = _normalize_non_blank(question, field_name="question")
    normalized_answer = _normalize_non_blank(answer, field_name="answer")
    if not supporting_candidates:
        return _no_evidence_result()

    try:
        raw = adapter.validate(
            question=normalized_question,
            answer=normalized_answer,
            supporting_candidates=supporting_candidates,
        )
    except GroundingValidationError:
        raise
    except Exception as error:
        raise GroundingValidationProviderError(
            "grounding validation provider failed"
        ) from error

    try:
        parsed = (
            raw
            if isinstance(raw, GroundingValidatorStructuredOutput)
            else GroundingValidatorStructuredOutput.model_validate(raw)
        )
    except ValidationError as error:
        raise GroundingValidationOutputError(
            "grounding validation output is malformed"
        ) from error

    active_settings = settings or get_settings()
    effective_grounded = (
        parsed.grounded
        and parsed.supported
        and not parsed.unsupported_claims
        and not parsed.missing_evidence
        and not parsed.ambiguous_claims
        and parsed.entailment_score >= active_settings.min_evidence_entailment_score
    )
    return GroundingValidationResult(
        grounded=effective_grounded,
        supported=parsed.supported,
        unsupported_claims=parsed.unsupported_claims,
        missing_evidence=parsed.missing_evidence,
        ambiguous_claims=parsed.ambiguous_claims,
        entailment_score=parsed.entailment_score,
    )


class OpenAIGroundingValidatorAdapter:
    """OpenAI-backed structured validator, constructed only during runtime wiring."""

    def __init__(self, *, api_key: str, model: str = "gpt-4o-mini", base_url: str = "") -> None:
        if not api_key or not api_key.strip():
            raise GroundingValidationInputError("api_key must not be blank")
        if not model or not model.strip():
            raise GroundingValidationInputError("model must not be blank")
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - installation concern.
            raise GroundingValidationProviderError("openai package is required") from error

        self._model = model.strip()
        # OpenAI SDK defaults to api.openai.com. base_url is empty by
        # default; an explicit value redirects the client at an alternate
        # OpenAI-compatible provider without changing the assignment default.
        client_kwargs: dict[str, object] = {"api_key": api_key}
        if base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        self._client = OpenAI(**client_kwargs)

    @classmethod
    def from_settings(
        cls, *, settings: Settings
    ) -> "OpenAIGroundingValidatorAdapter":
        """Construct the validator with the configured OpenAI model."""

        if settings.openai_api_key is None:
            raise GroundingValidationInputError("OPENAI_API_KEY must be configured")
        return cls(
            api_key=settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )

    def validate(
        self,
        *,
        question: str,
        answer: str,
        supporting_candidates: list[RetrievedCandidate],
    ) -> GroundingValidatorStructuredOutput:
        prompt = build_validation_prompt(
            question=question,
            answer=answer,
            supporting_candidates=supporting_candidates,
        )
        try:
            response = self._client.responses.parse(
                model=self._model,
                input=json.dumps({"validation_prompt": prompt}),
                text_format=GroundingValidatorStructuredOutput,
            )
        except Exception as error:  # pragma: no cover - live provider call.
            raise GroundingValidationProviderError(
                "grounding validation provider failed"
            ) from error

        if response.output_parsed is None:  # pragma: no cover - live provider call.
            raise GroundingValidationOutputError(
                "grounding validation output is malformed: missing parsed result"
            )
        return response.output_parsed
