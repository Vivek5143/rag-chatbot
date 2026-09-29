"""LangGraph orchestration for retrieval, evidence evaluation, and generation."""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.config import Settings
from src.generation import GenerationAdapter, generate_answer
from src.grading import EvidenceGraderAdapter, grade_evidence
from src.models import EvidenceAssessment, RetrievedCandidate
from src.retrieval import RetrievalService
from src.validation import (
    GroundingValidationResult,
    GroundingValidatorAdapter,
    validate_grounding,
)


REFUSAL_MESSAGE = (
    "I cannot answer this based on the information available in the provided document."
)


class GraphState(TypedDict, total=False):
    """Per-request state shared by the current and future workflow stages."""

    question: str
    retrieved_candidates: list[RetrievedCandidate]
    evidence_assessment: EvidenceAssessment
    supporting_candidates: list[RetrievedCandidate]
    answer: str | None
    validation_assessment: GroundingValidationResult
    confidence_score: float | None
    grounded: bool | None
    refused: bool


def _retrieve_node(
    state: GraphState, *, retrieval_service: RetrievalService
) -> GraphState:
    return {"retrieved_candidates": retrieval_service.retrieve(question=state["question"])}


def _evaluate_evidence_node(
    state: GraphState,
    *,
    grader_adapter: EvidenceGraderAdapter,
    settings: Settings | None,
) -> GraphState:
    candidates = state.get("retrieved_candidates", [])
    assessment = grade_evidence(
        question=state["question"],
        candidates=candidates,
        adapter=grader_adapter,
        settings=settings,
    )
    candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
    supporting_candidates = [
        candidates_by_id[chunk_id]
        for chunk_id in assessment.supporting_chunk_ids
    ]
    confidence_score = min(assessment.support_score, assessment.entailment_score)
    return {
        "evidence_assessment": assessment,
        "supporting_candidates": supporting_candidates,
        "confidence_score": confidence_score,
    }


def _route_after_evidence(state: GraphState) -> Literal["sufficient", "insufficient"]:
    assessment = state["evidence_assessment"]
    return "sufficient" if assessment.evidence_sufficient else "insufficient"


def _generate_node(
    state: GraphState, *, generation_adapter: GenerationAdapter
) -> GraphState:
    """Generate only from candidates selected by the evidence assessment."""

    result = generate_answer(
        question=state["question"],
        supporting_candidates=state["supporting_candidates"],
        adapter=generation_adapter,
    )
    return {
        "answer": result.answer,
        # Phase 7 must independently validate this generated answer.
        "grounded": False,
        "refused": False,
    }


def _refusal_node(state: GraphState) -> GraphState:
    del state
    return {
        "answer": REFUSAL_MESSAGE,
        "confidence_score": 0.0,
        "grounded": False,
        "refused": True,
    }


def _validate_node(
    state: GraphState,
    *,
    validator_adapter: GroundingValidatorAdapter,
    settings: Settings | None,
) -> GraphState:
    """Independently validate the generated answer against the same evidence."""

    assessment = validate_grounding(
        question=state["question"],
        answer=state["answer"],
        supporting_candidates=state["supporting_candidates"],
        adapter=validator_adapter,
        settings=settings,
    )
    return {
        "validation_assessment": assessment,
        "grounded": assessment.grounded,
        "refused": False,
    }


def _route_after_validation(state: GraphState) -> Literal["grounded", "ungrounded"]:
    return "grounded" if state["validation_assessment"].grounded else "ungrounded"


def build_graph(
    *,
    retrieval_service: RetrievalService,
    grader_adapter: EvidenceGraderAdapter,
    generation_adapter: GenerationAdapter,
    validator_adapter: GroundingValidatorAdapter,
    settings: Settings | None = None,
):
    """Build and compile the workflow; callers should reuse the compiled graph."""

    workflow = StateGraph(GraphState)
    workflow.add_node(
        "retrieve",
        lambda state: _retrieve_node(state, retrieval_service=retrieval_service),
    )
    workflow.add_node(
        "evaluate_evidence",
        lambda state: _evaluate_evidence_node(
            state, grader_adapter=grader_adapter, settings=settings
        ),
    )
    workflow.add_node(
        "generate",
        lambda state: _generate_node(state, generation_adapter=generation_adapter),
    )
    workflow.add_node(
        "validate",
        lambda state: _validate_node(
            state, validator_adapter=validator_adapter, settings=settings
        ),
    )
    workflow.add_node("refusal", _refusal_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "evaluate_evidence")
    workflow.add_conditional_edges(
        "evaluate_evidence",
        _route_after_evidence,
        {
            "sufficient": "generate",
            "insufficient": "refusal",
        },
    )
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validation,
        {
            "grounded": END,
            "ungrounded": "refusal",
        },
    )
    workflow.add_edge("refusal", END)
    return workflow.compile()
