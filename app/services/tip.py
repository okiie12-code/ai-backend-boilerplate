"""Deterministic verification for extracted tip commands.

This is the trust boundary. The model's output is treated as an untrusted
proposal; nothing here depends on the model having behaved. Every value is
re-checked against explicit rules, and anything that does not meet them is
rejected with a precise, machine-readable list of problems.
"""

from __future__ import annotations

import math
import re

from app.core.exceptions import AppError
from app.schemas.tip import TipCommand, VerifiedTip

ALLOWED_ACTIONS = frozenset({"tip", "send", "airdrop", "pay"})

# Assets this deployment is willing to move. A real system would load this from
# config or a chain registry; the point is that it is an allowlist, not "anything
# the model emitted".
KNOWN_TICKERS = frozenset({"SOL", "ETH", "MATIC", "USDC", "USDT", "BTC"})

MAX_AMOUNT = 1_000_000.0
# Platform handle rule: 2-32 chars, letters/digits/underscore. Tune per platform.
_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{2,32}$")


class TipValidationError(AppError):
    status_code = 422
    code = "tip_invalid"
    message = "The extracted command did not pass validation."


def normalize(command: TipCommand) -> TipCommand:
    """Canonicalise fields before checking them (lowercase action, strip @, upper ticker)."""
    return command.model_copy(
        update={
            "action": command.action.strip().lower(),
            "asset_ticker": command.asset_ticker.strip().upper().lstrip("$"),
            "target_user_id": command.target_user_id.strip().lstrip("@"),
        }
    )


def check(command: TipCommand) -> list[str]:
    """Return a list of problems with a (normalised) command. Empty list == valid."""
    problems: list[str] = []

    if command.action not in ALLOWED_ACTIONS:
        problems.append(f"action '{command.action}' is not one of {sorted(ALLOWED_ACTIONS)}")

    amount = command.amount
    if math.isnan(amount) or math.isinf(amount):
        problems.append("amount is not a finite number")
    elif amount <= 0:
        problems.append("amount must be greater than zero")
    elif amount > MAX_AMOUNT:
        problems.append(f"amount {amount} exceeds the per-command maximum of {MAX_AMOUNT}")

    if command.asset_ticker not in KNOWN_TICKERS:
        problems.append(
            f"asset_ticker '{command.asset_ticker}' is not a supported asset "
            f"({sorted(KNOWN_TICKERS)})"
        )

    if not _HANDLE_RE.match(command.target_user_id):
        problems.append("target_user_id must be 2-32 characters of letters, digits or underscore")

    return problems


def verify(command: TipCommand) -> VerifiedTip:
    """Normalise, validate, and return a VerifiedTip — or raise TipValidationError.

    This is the function the router calls before returning anything to the client.
    """
    normalized = normalize(command)
    problems = check(normalized)
    if problems:
        raise TipValidationError(
            details={"problems": problems, "normalized": normalized.model_dump()}
        )

    return VerifiedTip(
        action=normalized.action,
        amount=normalized.amount,
        is_fiat_denominated=normalized.is_fiat_denominated,
        asset_ticker=normalized.asset_ticker,
        target_user_id=normalized.target_user_id,
        # A fiat-denominated amount is valid but not yet settleable: it needs the
        # current price of asset_ticker before any on-chain amount can be computed.
        requires_price_conversion=normalized.is_fiat_denominated,
    )
