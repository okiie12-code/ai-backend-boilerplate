import asyncio
from decimal import Decimal

import pytest

from app.schemas.commands import ParsedCommand
from app.services.command_service import CommandError, CommandService, PlanNoLongerFeasible
from app.services.ledger import Ledger


class StubAI:
    """Returns a scripted ParsedCommand instead of calling the model."""

    def __init__(self, command: ParsedCommand) -> None:
        self.command = command

    async def extract(self, text: str, schema: type, **_: object) -> tuple[ParsedCommand, None]:
        return self.command, None


def make_service(command: ParsedCommand) -> CommandService:
    return CommandService(Ledger.demo(), StubAI(command), {})  # type: ignore[arg-type]


def cmd(**kw: object) -> ParsedCommand:
    base: dict[str, object] = {
        "action": "tip",
        "asset": "MATIC",
        "amount": 2.0,
        "distribution": "each",
        "recipients": [],
    }
    base.update(kw)
    return ParsedCommand.model_validate(base)


def test_resolve_each_named_recipients() -> None:
    service = make_service(cmd(recipients=["alice", "bob"], amount=2.0, distribution="each"))
    plan = service.resolve(service_command(service), sender="treasury")
    assert plan.total == Decimal("4")
    assert {t.recipient: t.amount for t in plan.transfers} == {
        "alice": Decimal("2"),
        "bob": Decimal("2"),
    }


def service_command(service: CommandService) -> ParsedCommand:
    return service._ai.command  # type: ignore[attr-defined]


def test_resolve_split_total_divides_evenly() -> None:
    service = make_service(
        cmd(
            action="airdrop",
            amount=9.0,
            distribution="split_total",
            recipients=["alice", "bob", "carol"],
        )
    )
    plan = service.resolve(service_command(service), sender="treasury")
    assert plan.total == Decimal("9")
    assert all(t.amount == Decimal("3") for t in plan.transfers)


def test_split_total_remainder_goes_to_first() -> None:
    service = make_service(
        cmd(amount=10.0, distribution="split_total", recipients=["alice", "bob", "carol"])
    )
    plan = service.resolve(service_command(service), sender="treasury")
    assert plan.total == Decimal("10")
    # 10 / 3 = 3.333333 each; first absorbs the rounding remainder.
    assert plan.transfers[0].amount == Decimal("3.333334")
    assert plan.transfers[1].amount == Decimal("3.333333")


def test_selector_resolves_top_active() -> None:
    service = make_service(
        cmd(
            action="airdrop",
            amount=6.0,
            distribution="split_total",
            recipients=[],
            recipient_selector="most active users",
            recipient_count=3,
        )
    )
    plan = service.resolve(service_command(service), sender="treasury")
    assert [t.recipient for t in plan.transfers] == ["alice", "bob", "carol"]


def test_unknown_recipient_rejected() -> None:
    service = make_service(cmd(recipients=["ghost"]))
    with pytest.raises(CommandError):
        service.resolve(service_command(service), sender="treasury")


def test_no_recipients_rejected() -> None:
    service = make_service(cmd(recipients=[]))
    with pytest.raises(CommandError):
        service.resolve(service_command(service), sender="treasury")


def test_sender_excluded_from_named_recipients() -> None:
    service = make_service(cmd(recipients=["treasury", "alice"]))
    plan = service.resolve(service_command(service), sender="treasury")
    assert [t.recipient for t in plan.transfers] == ["alice"]


def test_execute_applies_and_is_idempotent() -> None:
    service = make_service(cmd(recipients=["alice"], amount=5.0, distribution="each"))
    ledger: Ledger = service._ledger  # type: ignore[attr-defined]
    plan = service.resolve(service_command(service), sender="treasury")
    token = service.stage(plan)

    asyncio.run(service.execute(token))
    asyncio.run(service.execute(token))  # replay: must not double-pay
    assert ledger.get("alice").balance("MATIC") == Decimal("255")  # 250 + 5 once


def test_feasibility_flags_overdraft() -> None:
    service = make_service(cmd(recipients=["alice"], amount=1000.0, distribution="each"))
    plan = service.resolve(service_command(service), sender="dave")
    feasible, reason = service.feasibility(plan)
    assert feasible is False
    assert reason is not None


def test_execute_rechecks_balance_at_apply_time() -> None:
    """A plan that was feasible at preview fails cleanly if funds are gone by execute."""
    service = make_service(cmd(recipients=["alice"], amount=8.0, distribution="each"))
    ledger: Ledger = service._ledger  # type: ignore[attr-defined]
    plan = service.resolve(service_command(service), sender="dave")  # dave has 10
    token = service.stage(plan)

    # Drain dave before execution.
    asyncio.run(
        ledger.apply(
            sender="dave", asset="MATIC", transfers=[("bob", Decimal("5"))], idempotency_key="drain"
        )
    )
    with pytest.raises(PlanNoLongerFeasible):
        asyncio.run(service.execute(token))
