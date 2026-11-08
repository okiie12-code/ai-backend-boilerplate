import asyncio
from typing import Any

import anthropic
import httpx2
import pytest

from app.core.config import Settings
from app.core.exceptions import (
    LLMBadResponseError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.schemas.ai import LeadExtraction
from app.services.ai_service import AIService, map_provider_error
from tests.conftest import make_message

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[anthropic.APIStatusError], code: int, **headers: str) -> Any:
    response = httpx2.Response(code, request=REQUEST, headers=headers)
    return cls("error", response=response, body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (anthropic.APITimeoutError(request=REQUEST), LLMTimeoutError),
        (anthropic.APIConnectionError(request=REQUEST), LLMUnavailableError),
        (status_error(anthropic.AuthenticationError, 401), LLMConfigurationError),
        (status_error(anthropic.InternalServerError, 529), LLMUnavailableError),
    ],
)
def test_map_provider_error(exc: Exception, expected: type[Exception]) -> None:
    assert isinstance(map_provider_error(exc), expected)


def test_rate_limit_keeps_retry_after() -> None:
    mapped = map_provider_error(
        status_error(anthropic.RateLimitError, 429, **{"retry-after": "12"})
    )
    assert isinstance(mapped, LLMRateLimitError)
    assert mapped.retry_after == 12


class FakeMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: Any) -> None:
        self.messages = FakeMessages(result)


def make_service(result: Any, **overrides: Any) -> tuple[AIService, FakeClient]:
    client = FakeClient(result)
    settings = Settings(anthropic_api_key="sk-test", **overrides)
    return AIService(client, settings), client  # type: ignore[arg-type]


VALID_LEAD = (
    '{"full_name": "Jane Doe", "email": "jane@acme.io", "company": "Acme", '
    '"budget_usd": 20000, "intent": "demo", "urgency": "high", "summary": "Wants a demo."}'
)


def test_extract_validates_and_sends_schema() -> None:
    service, client = make_service(make_message(VALID_LEAD))
    lead, _ = asyncio.run(service.extract("text", LeadExtraction, instructions="x"))
    assert lead.company == "Acme"

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["effort"] == "medium"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["extra_body"] == {"fallbacks": "default"}


def test_extract_fallback_can_be_disabled() -> None:
    service, client = make_service(make_message(VALID_LEAD), llm_refusal_fallback=False)
    asyncio.run(service.extract("text", LeadExtraction, instructions="x"))
    assert "extra_body" not in client.messages.calls[0]


def test_extract_rejects_constraint_violations() -> None:
    bad = VALID_LEAD.replace("20000", "-5")  # budget_usd has ge=0
    service, _ = make_service(make_message(bad))
    with pytest.raises(LLMBadResponseError):
        asyncio.run(service.extract("text", LeadExtraction, instructions="x"))


def test_extract_refusal() -> None:
    service, _ = make_service(make_message("", stop_reason="refusal"))
    with pytest.raises(LLMRefusalError):
        asyncio.run(service.extract("text", LeadExtraction, instructions="x"))


def test_extract_maps_sdk_errors() -> None:
    service, _ = make_service(status_error(anthropic.RateLimitError, 429))
    with pytest.raises(LLMRateLimitError):
        asyncio.run(service.extract("text", LeadExtraction, instructions="x"))


def test_missing_credentials_maps_to_configuration_error() -> None:
    # The SDK raises this TypeError at request-build time when no key resolves.
    err = TypeError("Could not resolve authentication method. Expected one of api_key...")
    service, _ = make_service(err)
    with pytest.raises(LLMConfigurationError):
        asyncio.run(service.extract("text", LeadExtraction, instructions="x"))
