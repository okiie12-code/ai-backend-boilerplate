import json

from fastapi.testclient import TestClient

from app.core.exceptions import LLMRateLimitError, LLMUnavailableError
from app.schemas.ai import LeadExtraction, StreamEvent
from tests.conftest import FakeAIService, done_event


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_request_id_is_propagated(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id\r\n"})
    assert response.headers["X-Request-ID"] != "bad id\r\n"


def test_validation_error_shape(client: TestClient) -> None:
    response = client.post("/api/v1/ai/chat/stream", json={"prompt": ""})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["request_id"] == response.headers["X-Request-ID"]


def test_chat_stream_emits_sse(client: TestClient, fake_service: FakeAIService) -> None:
    fake_service.stream_events = [
        StreamEvent(type="start", model="claude-opus-5-5"),
        StreamEvent(type="delta", text="Hello"),
        StreamEvent(type="delta", text=" world"),
        done_event(),
    ]
    response = client.post("/api/v1/ai/chat/stream", json={"prompt": "hi"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["start", "delta", "delta", "done"]
    assert "".join(str(d.get("text", "")) for _, d in events) == "Hello world"


def test_rate_limit_before_stream_is_http_429(
    client: TestClient, fake_service: FakeAIService
) -> None:
    fake_service.stream_events = [LLMRateLimitError(retry_after=7)]
    response = client.post("/api/v1/ai/chat/stream", json={"prompt": "hi"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "7"
    assert response.json()["error"]["code"] == "llm_rate_limited"


def test_error_mid_stream_becomes_sse_error_event(
    client: TestClient, fake_service: FakeAIService
) -> None:
    fake_service.stream_events = [
        StreamEvent(type="start", model="claude-opus-5-5"),
        StreamEvent(type="delta", text="partial"),
        LLMUnavailableError(),
    ]
    response = client.post("/api/v1/ai/chat/stream", json={"prompt": "hi"})
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["code"] == "llm_unavailable"  # type: ignore[index]


def test_extract_lead(client: TestClient, fake_service: FakeAIService) -> None:
    fake_service.extract_result = LeadExtraction(
        full_name="Jane Doe",
        email="jane@acme.io",
        company="Acme",
        budget_usd=20000,
        intent="demo",
        urgency="high",
        summary="Wants a demo before Q1.",
    )
    response = client.post("/api/v1/ai/extract/lead", json={"text": "messy text"})
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["email"] == "jane@acme.io"
    assert body["usage"] == {"input_tokens": 12, "output_tokens": 34}


def test_unhandled_error_is_generic_500(client: TestClient, fake_service: FakeAIService) -> None:
    fake_service.extract_error = RuntimeError("secret internals")
    response = client.post("/api/v1/ai/extract/lead", json={"text": "x"})
    assert response.status_code == 500
    assert "secret" not in response.text
    assert response.json()["error"]["code"] == "internal_error"
