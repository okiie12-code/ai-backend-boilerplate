"""Route test for /tip/extract with the LLM parse stubbed.

Confirms the happy path returns the verified payload and that a payload failing
deterministic validation comes back as 422 tip_invalid.
"""

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_ai_service
from app.core.config import Settings
from app.main import create_app
from app.schemas.tip import TipCommand
from tests.conftest import make_message


class StubExtractAI:
    def __init__(self, command: TipCommand) -> None:
        self.command = command

    async def extract(self, text: str, schema: type, **_: object) -> tuple[TipCommand, object]:
        return self.command, make_message("{}")


def make_client(command: TipCommand) -> TestClient:
    app = create_app(Settings(environment="local", anthropic_api_key="sk-test"))
    app.dependency_overrides[get_ai_service] = lambda: StubExtractAI(command)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def valid_cmd() -> TipCommand:
    return TipCommand(
        action="tip",
        amount=10.0,
        is_fiat_denominated=True,
        asset_ticker="SOL",
        target_user_id="crek",
    )


def test_extract_tip_happy_path(valid_cmd: TipCommand) -> None:
    client = make_client(valid_cmd)
    with client:
        r = client.post("/api/v1/tip/extract", json={"text": "give ten dollars of Solana to @crek"})
    assert r.status_code == 200
    body = r.json()
    assert body["verified"] == {
        "action": "tip",
        "amount": 10.0,
        "is_fiat_denominated": True,
        "asset_ticker": "SOL",
        "target_user_id": "crek",
        "requires_price_conversion": True,
    }


def test_extract_tip_rejects_unknown_asset() -> None:
    bad = TipCommand(
        action="tip",
        amount=10.0,
        is_fiat_denominated=False,
        asset_ticker="DOGE",
        target_user_id="crek",
    )
    client = make_client(bad)
    with client:
        r = client.post("/api/v1/tip/extract", json={"text": "tip 10 DOGE to @crek"})
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == "tip_invalid"
    assert any("not a supported asset" in p for p in error["details"]["problems"])
