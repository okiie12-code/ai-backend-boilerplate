"""Route-level test of the full preview -> execute flow against the real demo ledger.

Only the LLM parse step is replaced; resolution, feasibility, token staging,
execution and idempotency all run for real.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_command_service
from app.core.config import Settings
from app.main import create_app
from app.schemas.commands import ParsedCommand
from app.services.command_service import CommandService
from app.services.ledger import Ledger


@pytest.fixture
def parsed() -> dict[str, ParsedCommand]:
    return {
        "value": ParsedCommand(
            action="airdrop",
            asset="MATIC",
            amount=6.0,
            distribution="split_total",
            recipients=[],
            recipient_selector="most active users",
            recipient_count=3,
        )
    }


@pytest.fixture
def client(parsed: dict[str, ParsedCommand]) -> Iterator[TestClient]:
    app = create_app(Settings(environment="local", anthropic_api_key="sk-test"))

    # Shared ledger + pending store so preview and execute see the same state.
    ledger = Ledger.demo()
    pending: dict[str, object] = {}

    class StubParseAI:
        async def extract(self, text: str, schema: type, **_: object) -> tuple[ParsedCommand, None]:
            return parsed["value"], None

    def override() -> CommandService:
        return CommandService(ledger, StubParseAI(), pending)  # type: ignore[arg-type]

    app.state.ledger = ledger
    app.dependency_overrides[get_command_service] = override
    with TestClient(app, raise_server_exceptions=False) as c:
        c.app.state.ledger = ledger  # type: ignore[attr-defined]
        yield c


def test_preview_then_execute_moves_funds(client: TestClient) -> None:
    preview = client.post("/api/v1/commands/preview", json={"text": "airdrop 6 MATIC to top 3"})
    assert preview.status_code == 200
    body = preview.json()
    assert body["feasible"] is True
    assert [t["recipient"] for t in body["plan"]["transfers"]] == ["alice", "bob", "carol"]
    token = body["confirmation_token"]
    assert token

    before = _balance(client, "alice")
    execute = client.post("/api/v1/commands/execute", json={"confirmation_token": token})
    assert execute.status_code == 200
    assert execute.json()["executed"] is True
    assert _balance(client, "alice") == before + 2.0  # 6 split across 3


def test_execute_is_idempotent_over_http(client: TestClient) -> None:
    token = client.post(
        "/api/v1/commands/preview", json={"text": "airdrop 6 MATIC to top 3"}
    ).json()["confirmation_token"]
    client.post("/api/v1/commands/execute", json={"confirmation_token": token})
    before = _balance(client, "alice")
    # Replaying the same token must not pay again.
    again = client.post("/api/v1/commands/execute", json={"confirmation_token": token})
    assert again.status_code == 200
    assert _balance(client, "alice") == before


def test_unknown_token_is_404(client: TestClient) -> None:
    response = client.post("/api/v1/commands/execute", json={"confirmation_token": "nope"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_token"


def _balance(client: TestClient, user: str) -> float:
    accounts = client.get("/api/v1/commands/ledger").json()["accounts"]
    account = next(a for a in accounts if a["username"] == user)
    return float(account["balances"]["MATIC"])
