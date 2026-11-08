from collections.abc import AsyncGenerator, Iterator
from typing import Any

import anthropic
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ai_service
from app.core.config import Settings
from app.main import create_app
from app.schemas.ai import StreamEvent, Usage


def make_message(text: str, stop_reason: str = "end_turn") -> anthropic.types.Message:
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 34},
        }
    )


class FakeAIService:
    """Stands in for AIService in router tests. Set the attributes to script behaviour."""

    def __init__(self) -> None:
        self.stream_events: list[StreamEvent | Exception] = []
        self.extract_result: Any = None
        self.extract_error: Exception | None = None

    async def stream_chat(self, prompt: str, **_: Any) -> AsyncGenerator[StreamEvent, None]:
        for item in self.stream_events:
            if isinstance(item, Exception):
                raise item
            yield item

    async def extract(self, text: str, schema: Any, **_: Any) -> Any:
        if self.extract_error:
            raise self.extract_error
        return self.extract_result, make_message("{}")


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="local", anthropic_api_key="sk-test", log_level="WARNING")


@pytest.fixture
def fake_service() -> FakeAIService:
    return FakeAIService()


@pytest.fixture
def client(settings: Settings, fake_service: FakeAIService) -> Iterator[TestClient]:
    app = create_app(settings)
    app.dependency_overrides[get_ai_service] = lambda: fake_service
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def done_event() -> StreamEvent:
    return StreamEvent(
        type="done",
        model="claude-opus-5-5",
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=2),
    )
