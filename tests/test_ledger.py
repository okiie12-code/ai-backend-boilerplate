import asyncio
from decimal import Decimal

import pytest

from app.services.ledger import InsufficientFunds, Ledger, UnknownAccount


def test_demo_world_has_balances() -> None:
    ledger = Ledger.demo()
    assert ledger.get("alice").balance("MATIC") == Decimal("250")
    assert ledger.get("dave").balance("USDC") == Decimal("0")


def test_top_active_excludes_sender_and_ranks_by_score() -> None:
    ledger = Ledger.demo()
    assert ledger.top_active(3, exclude="treasury") == ["alice", "bob", "carol"]


def test_apply_moves_funds() -> None:
    ledger = Ledger.demo()
    receipt = asyncio.run(
        ledger.apply(
            sender="treasury",
            asset="MATIC",
            transfers=[("alice", Decimal("10")), ("bob", Decimal("5"))],
            idempotency_key="k1",
        )
    )
    assert receipt.total == Decimal("15")
    assert ledger.get("alice").balance("MATIC") == Decimal("260")
    assert ledger.get("treasury").balance("MATIC") == Decimal("99985")


def test_apply_is_idempotent_per_key() -> None:
    ledger = Ledger.demo()
    kwargs = dict(sender="treasury", asset="MATIC", transfers=[("alice", Decimal("10"))])
    asyncio.run(ledger.apply(idempotency_key="same", **kwargs))  # type: ignore[arg-type]
    asyncio.run(ledger.apply(idempotency_key="same", **kwargs))  # type: ignore[arg-type]
    # Applied once despite two calls.
    assert ledger.get("alice").balance("MATIC") == Decimal("260")


def test_insufficient_funds_rejected() -> None:
    ledger = Ledger.demo()
    with pytest.raises(InsufficientFunds):
        asyncio.run(
            ledger.apply(
                sender="dave",
                asset="MATIC",
                transfers=[("alice", Decimal("1000"))],
                idempotency_key="k",
            )
        )
    assert ledger.get("dave").balance("MATIC") == Decimal("10")  # unchanged


def test_concurrent_claims_cannot_both_spend() -> None:
    """Two requests for more than the balance: exactly one wins, no overdraft."""
    ledger = Ledger.demo()

    # dave has 10 MATIC; two transfers of 8 each cannot both succeed.
    async def run() -> list[object]:
        return await asyncio.gather(
            ledger.apply(
                sender="dave",
                asset="MATIC",
                transfers=[("alice", Decimal("8"))],
                idempotency_key="a",
            ),
            ledger.apply(
                sender="dave", asset="MATIC", transfers=[("bob", Decimal("8"))], idempotency_key="b"
            ),
            return_exceptions=True,
        )

    results = asyncio.run(run())
    failures = [r for r in results if isinstance(r, InsufficientFunds)]
    assert len(failures) == 1
    assert ledger.get("dave").balance("MATIC") == Decimal("2")  # only one 8 left


def test_unknown_account() -> None:
    ledger = Ledger.demo()
    with pytest.raises(UnknownAccount):
        asyncio.run(
            ledger.apply(
                sender="treasury",
                asset="MATIC",
                transfers=[("nobody", Decimal("1"))],
                idempotency_key="k",
            )
        )
