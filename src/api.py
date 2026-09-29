"""FastAPI composition layer for the compiled, validated RAG graph."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping, Protocol

from fastapi import Depends, FastAPI, HTTPException, status

from src.config import ConfigurationError, Settings, get_settings
from src.embeddings import EmbeddingConfig, EmbeddingError, EmbeddingService, OpenAIEmbeddingProvider
from src.generation import GenerationError, OpenAIGenerationAdapter
from src.graph import GraphState, REFUSAL_MESSAGE, build_graph
from src.grading import EvidenceGraderError, OpenAIEvidenceGraderAdapter
from src.indexing import PineconeConfigurationError, PineconeProviderError, PineconeVectorIndex
from src.models import ChatRequest, ChatResponse, HealthResponse, RetrievedCandidate, RetrievedContext
from src.retrieval import RetrievalError, RetrievalService
from src.validation import GroundingValidationError, OpenAIGroundingValidatorAdapter


SERVICE_UNAVAILABLE_DETAIL = "RAG service is currently unavailable."


class CompiledGraph(Protocol):
    """Small graph port that keeps FastAPI tests independent of LangGraph."""

    def invoke(self, input: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute one graph request and return its final state."""


GraphFactory = Callable[[], CompiledGraph]


class GraphResponseError(RuntimeError):
    """Raised when a graph result cannot safely satisfy the public contract."""


def build_runtime_graph() -> CompiledGraph:
    """Construct the production graph without contacting providers at import time."""

    settings = get_settings()
    settings.require_retrieval_configuration()
    assert settings.openai_api_key is not None  # Proven by the check above.
    assert settings.pinecone_api_key is not None

    api_key = settings.openai_api_key.get_secret_value()
    embeddings = EmbeddingService(
        OpenAIEmbeddingProvider(api_key=api_key, base_url=settings.openai_base_url),
        EmbeddingConfig(
            model=settings.openai_embedding_model,
            dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
        ),
    )
    vector_index = PineconeVectorIndex.from_api_key(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index_name,
    )
    retrieval_service = RetrievalService(
        embeddings=embeddings,
        vector_index=vector_index,
        namespace=settings.pinecone_namespace,
        top_k=settings.top_k,
    )
    grader_adapter = OpenAIEvidenceGraderAdapter(
        api_key=api_key,
        model=settings.openai_model,
        timeout_seconds=settings.evidence_grader_timeout_seconds,
        base_url=settings.openai_base_url,
    )

    return build_graph(
        retrieval_service=retrieval_service,
        grader_adapter=grader_adapter,
        generation_adapter=OpenAIGenerationAdapter.from_settings(settings=settings),
        validator_adapter=OpenAIGroundingValidatorAdapter.from_settings(settings=settings),
        settings=settings,
    )


def _retrieved_context(
    candidates: object,
    *,
    context_limit: int,
) -> list[RetrievedContext]:
    if candidates is None:
        return []
    if not isinstance(candidates, list):
        raise GraphResponseError("retrieved candidates are invalid")

    context: list[RetrievedContext] = []
    for candidate in candidates[:context_limit]:
        if not isinstance(candidate, RetrievedCandidate):
            raise GraphResponseError("retrieved candidate is invalid")
        context.append(
            RetrievedContext(
                chunk_id=candidate.chunk_id,
                source=candidate.source,
                page=candidate.page,
                content=candidate.content,
                source_hash=candidate.source_hash,
                similarity_score=candidate.similarity_score,
            )
        )
    return context


def _refusal_reason(state: Mapping[str, Any]) -> str:
    assessment = state.get("evidence_assessment")
    if getattr(assessment, "evidence_sufficient", None) is False:
        return str(getattr(assessment, "reason", "insufficient_evidence"))
    return "grounding_validation_failed"


def response_from_graph_state(
    state: Mapping[str, Any],
    *,
    context_limit: int,
) -> ChatResponse:
    """Map only a complete graph terminal state into the public response model."""

    answer = state.get("answer")
    grounded = state.get("grounded")
    refused = state.get("refused")
    confidence_score = state.get("confidence_score")

    if not isinstance(answer, str) or not isinstance(grounded, bool):
        raise GraphResponseError("graph result is incomplete")
    if not isinstance(refused, bool) or not isinstance(confidence_score, (int, float)):
        raise GraphResponseError("graph result is invalid")

    if refused:
        if answer != REFUSAL_MESSAGE or grounded or confidence_score != 0.0:
            raise GraphResponseError("refusal state is invalid")
        refusal_reason: str | None = _refusal_reason(state)
    elif not grounded:
        raise GraphResponseError("non-refusal result is not grounded")
    else:
        refusal_reason = None

    return ChatResponse(
        final_answer=answer,
        retrieved_context=_retrieved_context(
            state.get("retrieved_candidates"), context_limit=context_limit
        ),
        confidence_score=float(confidence_score),
        grounded=grounded,
        refusal_reason=refusal_reason,
    )


def _service_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=SERVICE_UNAVAILABLE_DETAIL,
    )


def create_app(
    *,
    graph_factory: GraphFactory = build_runtime_graph,
    settings_factory: Callable[[], Settings] = get_settings,
    context_limit: int | None = None,
) -> FastAPI:
    """Create a testable API whose graph is compiled lazily once per app instance.

    ``context_limit`` caps the number of retrieved candidates returned in the
    response envelope. It defaults to the configured retrieval ``top_k`` so the
    API respects the existing configuration by default, while remaining
    overridable for tests and future response-shaping needs.
    """

    app = FastAPI(
        title="Agentic AI eBook RAG Chatbot",
        version="0.8.0",
        description="Validated document-grounded RAG API.",
    )

    @lru_cache
    def get_graph() -> CompiledGraph:
        return graph_factory()

    def graph_dependency() -> CompiledGraph:
        try:
            return get_graph()
        except (
            ConfigurationError,
            EmbeddingError,
            PineconeConfigurationError,
            PineconeProviderError,
            RetrievalError,
            EvidenceGraderError,
            GenerationError,
            GroundingValidationError,
        ) as error:
            raise _service_unavailable() from error
        except Exception as error:
            raise _service_unavailable() from error

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Report process liveness without embedding, retrieval, or provider calls."""

        return HealthResponse(status="ok", phase="api")

    @app.post("/chat", response_model=ChatResponse, tags=["chat"])
    def chat(
        request: ChatRequest,
        graph: CompiledGraph = Depends(graph_dependency),
    ) -> ChatResponse:
        """Run the single compiled graph path and safely serialize its terminal state."""

        try:
            state = graph.invoke({"question": request.query})
            settings = settings_factory()
            limit = context_limit if context_limit is not None else settings.top_k
            return response_from_graph_state(state, context_limit=limit)
        except HTTPException:
            raise
        except (
            ConfigurationError,
            EmbeddingError,
            PineconeConfigurationError,
            PineconeProviderError,
            RetrievalError,
            EvidenceGraderError,
            GenerationError,
            GroundingValidationError,
            GraphResponseError,
        ) as error:
            raise _service_unavailable() from error
        except Exception as error:
            raise _service_unavailable() from error

    return app
