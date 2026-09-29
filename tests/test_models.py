import pytest
from pydantic import ValidationError

from src.models import ChatRequest, ChatResponse


def test_chat_request_normalizes_valid_query() -> None:
    request = ChatRequest(query="  What is Agentic AI?  ")

    assert request.query == "What is Agentic AI?"


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_chat_request_rejects_blank_query(query: str) -> None:
    with pytest.raises(ValidationError, match="query must not be blank"):
        ChatRequest(query=query)


def test_chat_response_rejects_nonzero_confidence_when_ungrounded() -> None:
    with pytest.raises(ValidationError, match="ungrounded responses"):
        ChatResponse(
            final_answer="Unsupported answer",
            confidence_score=0.3,
            grounded=False,
        )
