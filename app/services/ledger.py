"""A mock, in-memory ledger — a stand-in for a real custodial database.

This exists to demonstrate the natural-language-to-transaction flow **safely**:
no real funds, no external calls, no persistence. The only thing it takes
seriously is the invariant that matters in a real money system — a balance is
never debited below zero, and the same transaction is never applied twice.

The LLM never touches this. It only ever proposes a structured command; every
mutation here goes through deterministic, validated code under a lock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal

Asset = str
PRECISION = Decimal("0.000001")


def quantize(amount: Decimal) -> Decimal:
    return amount.quantize(PRECISION)


class LedgerError(Exception):
    """Base class for ledger failures."""


class UnknownAccount(LedgerError):
    pass


class InsufficientFunds(LedgerError):
    def __init__(self, username: str, asset: Asset, needed: Decimal, available: Decimal) -> None:
        super().__init__(f"{username} has {available} {asset}, needs {needed} {asset}")
        self.username = username
        self.asset = asset
        self.needed = needed
        self.available = available


@dataclass(frozen=True)
class Receipt:
    idempotency_key: str
    sender: str
    asset: Asset
    transfers: tuple[tuple[str, Decimal], ...]  # (recipient, amount)
    total: Decimal


@dataclass
class Account:
    username: str
    balances: dict[Asset, Decimal] = field(default_factory=dict)
    activity_score: int = 0  # drives the "most active users" selector demo

    def balance(self, asset: Asset) -> Decimal:
        return self.balances.get(asset, Decimal(0))


class Ledger:
    """Async-safe in-memory ledger. One instance lives on `app.state.ledger`."""

    def __init__(self) -> None:
        self._accounts: dict[str, Account] = {}
        self._lock = asyncio.Lock()
        self._applied: dict[str, Receipt] = {}  # idempotency: key -> receipt

    # -- construction -----------------------------------------------------

    @classmethod
    def demo(cls) -> Ledger:
        """A small world of accounts so the endpoints return something interesting."""
        ledger = cls()
        seed = [
            ("treasury", {"MATIC": "100000", "USDC": "50000", "ETH": "50"}, 0),
            ("alice", {"MATIC": "250", "USDC": "40"}, 93),
            ("bob", {"MATIC": "120", "USDC": "15"}, 71),
            ("carol", {"MATIC": "60"}, 64),
            ("dave", {"MATIC": "10"}, 22),
        ]
        for username, balances, activity in seed:
            ledger._accounts[username] = Account(
                username=username,
                balances={a: Decimal(v) for a, v in balances.items()},
                activity_score=activity,
            )
        return ledger

    # -- reads ------------------------------------------------------------

    def get(self, username: str) -> Account:
        account = self._accounts.get(username)
        if account is None:
            raise UnknownAccount(username)
        return account

    def exists(self, username: str) -> bool:
        return username in self._accounts

    def accounts(self) -> list[Account]:
        return list(self._accounts.values())

    def top_active(self, count: int, *, exclude: str | None = None) -> list[str]:
        """Resolve a selector like 'the N most active users' deterministically."""
        ranked = sorted(
            (a for a in self._accounts.values() if a.username != exclude),
            key=lambda a: (-a.activity_score, a.username),
        )
        return [a.username for a in ranked[:count]]

    # -- the one mutating operation --------------------------------------

    async def apply(
        self,
        *,
        sender: str,
        asset: Asset,
        transfers: list[tuple[str, Decimal]],
        idempotency_key: str,
    ) -> Receipt:
        """Atomically debit `sender` and credit each recipient.

        Serialised by a lock and re-checked against live balances, so two
        concurrent requests for the same funds cannot both succeed. Replaying
        the same `idempotency_key` returns the original receipt instead of
        moving funds again — the guard against double-spend on retry.
        """
        async with self._lock:
            if (existing := self._applied.get(idempotency_key)) is not None:
                return existing

            sender_account = self.get(sender)
            normalised = [(r, quantize(amount)) for r, amount in transfers]
            for recipient, _ in normalised:
                if not self.exists(recipient):
                    raise UnknownAccount(recipient)

            total = quantize(sum((amount for _, amount in normalised), Decimal(0)))
            available = sender_account.balance(asset)
            if total > available:
                raise InsufficientFunds(sender, asset, total, available)

            sender_account.balances[asset] = quantize(available - total)
            for recipient, amount in normalised:
                account = self.get(recipient)
                account.balances[asset] = quantize(account.balance(asset) + amount)

            receipt = Receipt(
                idempotency_key=idempotency_key,
                sender=sender,
                asset=asset,
                transfers=tuple(normalised),
                total=total,
            )
            self._applied[idempotency_key] = receipt
            return receipt
