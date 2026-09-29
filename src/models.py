"""Minimal shared domain and API models for the Phase 1 foundation."""

from __future__ import annotations

from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


MAX_QUERY_CHARACTERS = 2000


class ChatRequest(BaseModel):
    """Validated input for the chat endpoint."""

    query: str = Field(..., max_length=MAX_QUERY_CHARACTERS)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RetrievedContext(BaseModel):
    """API-safe retrieved evidence with provenance for inspection."""

    chunk_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    page: int | None = Field(default=None, ge=1)
    content: str = Field(..., min_length=1)
    source_hash: str = Field(..., min_length=1)
    similarity_score: float


class RetrievedCandidate(BaseModel):
    """Internal retrieval candidate with unmodified provenance and raw similarity."""

    chunk_id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    page: int = Field(..., ge=1)
    source_hash: str = Field(..., min_length=1)
    similarity_score: float
    metadata: dict[str, Any]

    @field_validator("similarity_score")
    @classmethod
    def require_finite_similarity_score(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("similarity_score must be finite")
        return value


class EvidenceAssessment(BaseModel):
    """Structured evidence gate used by later graph and validation phases."""

    relevant: bool
    complete: bool
    support_score: float = Field(ge=0.0, le=1.0)
    entailment_score: float = Field(ge=0.0, le=1.0)
    evidence_sufficient: bool
    reason: str = Field(..., min_length=1)
    supporting_chunk_ids: list[str]
    missing_aspects: list[str]

    @field_validator("support_score", "entailment_score")
    @classmethod
    def require_finite_evidence_scores(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("evidence scores must be finite")
        return value

    @model_validator(mode="after")
    def validate_sufficient_evidence(self) -> "EvidenceAssessment":
        if self.evidence_sufficient and not (self.relevant and self.complete):
            raise ValueError("sufficient evidence must be both relevant and complete")
        if self.evidence_sufficient and not self.supporting_chunk_ids:
            raise ValueError("sufficient evidence must identify supporting chunks")
        if self.evidence_sufficient and self.missing_aspects:
            raise ValueError("sufficient evidence must not have missing aspects")
        return self


class ChatResponse(BaseModel):
    """Stable response envelope for future grounded answers and refusals."""

    final_answer: str = Field(..., min_length=1)
    retrieved_context: list[RetrievedContext] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    grounded: bool
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def validate_grounding_state(self) -> "ChatResponse":
        if not self.grounded and self.confidence_score != 0.0:
            raise ValueError("ungrounded responses must have zero confidence")
        if self.grounded and self.confidence_score == 0.0:
            raise ValueError("grounded responses must have non-zero evidence confidence")
        return self


class HealthResponse(BaseModel):
    """Lightweight process-liveness response for the API service."""

    status: Literal["ok"]
    phase: Literal["api"]
