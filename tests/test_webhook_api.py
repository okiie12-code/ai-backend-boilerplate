"""Full adapter flow: webhook event -> parse -> settle -> forward to mock core bot.

Only the model parse is stubbed; settlement validation and forwarding run for real.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ai_service, get_core_bot
from app.core.config import Settings
from app.main import create_app
from app.schemas.dwallet_intent import ParsedIntent
from app.services.core_bot import MockCoreBot
from tests.conftest import make_message


def make_client(command: ParsedIntent) -> tuple[TestClient, MockCoreBot]:
    app = create_app(Settings(environment="local", anthropic_api_key="sk-test"))
    # Override the dependency so this exact mock is used (lifespan startup sets its own).
    core_bot = MockCoreBot()

    class StubAI:
        async def extract(
            self, text: str, schema: type, **_: object
        ) -> tuple[ParsedIntent, object]:
            return command, make_message("{}")

    app.dependency_overrides[get_ai_service] = lambda: StubAI()
    app.dependency_overrides[get_core_bot] = lambda: core_bot
    return TestClient(app, raise_server_exceptions=False), core_bot


@pytest.fixture
def valid() -> ParsedIntent:
    return ParsedIntent(
        action="tip", amount=5.0, is_fiat_denominated=False, asset="SOL", recipient="crek"
    )


def test_webhook_forwards_clean_payload(valid: ParsedIntent) -> None:
    client, core_bot = make_client(valid)
    with client:
        r = client.post(
            "/api/v1/webhook/chat",
            json={"platform": "kick", "user_id": "u123", "text": "yo bot throw 5 sol to @crek"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["platform"] == "kick"
    assert body["forwarded"]["asset"] == "SOL"
    assert body["receipt"]["accepted"] is True
    assert body["receipt"]["tx_id"].startswith("mocktx_")
    # The core bot received exactly one clean settlement.
    assert len(core_bot.received) == 1
    assert core_bot.received[0].recipient == "crek"


def test_webhook_rejects_invalid_intent_before_forwarding() -> None:
    bad = ParsedIntent(
        action="tip", amount=-5.0, is_fiat_denominated=False, asset="SOL", recipient="crek"
    )
    client, core_bot = make_client(bad)
    with client:
        r = client.post(
            "/api/v1/webhook/chat",
            json={"platform": "discord", "user_id": "u1", "text": "tip -5 sol to @crek"},
        )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "intent_rejected"
    # Nothing was forwarded to the core bot.
    assert core_bot.received == []


def test_webhook_rejects_unknown_platform(valid: ParsedIntent) -> None:
    client, _ = make_client(valid)
    with client:
        r = client.post(
            "/api/v1/webhook/chat",
            json={"platform": "telegram", "user_id": "u1", "text": "hi"},
        )
    assert r.status_code == 422  # ChatWebhookEvent.platform is a Literal
