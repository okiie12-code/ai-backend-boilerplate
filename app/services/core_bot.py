"""The 'core bot' the adapter forwards clean payloads to.

In a real integration this is the bot's existing transaction API (an HTTP call to
Crek's endpoint). Here it is a Protocol with an in-process mock, so the whole
adapter flow can be demonstrated and tested without touching any real system.

Swapping the mock for `HttpCoreBot(base_url=...)` is the only change needed to
point the adapter at a real backend — the adapter code does not change.
"""

from __future__ import annotations

import uuid
from typing import Protocol

from app.core.logger import get_logger
from app.schemas.dwallet_intent import SettlementRequest
from app.schemas.webhook import CoreBotReceipt

log = get_logger(__name__)


class CoreBot(Protocol):
    async def submit(self, settlement: SettlementRequest) -> CoreBotReceipt: ...


class MockCoreBot:
    """Stand-in for the real transaction engine. Accepts only clean settlements."""

    def __init__(self) -> None:
        self.received: list[SettlementRequest] = []

    async def submit(self, settlement: SettlementRequest) -> CoreBotReceipt:
        # By the time anything reaches here it is already a validated
        # SettlementRequest — the core never parses raw text itself.
        self.received.append(settlement)
        tx_id = f"mocktx_{uuid.uuid4().hex[:12]}"
        log.info(
            "core_bot.accepted",
            tx_id=tx_id,
            action=settlement.action,
            asset=settlement.asset,
            amount=settlement.amount,
        )
        return CoreBotReceipt(accepted=True, tx_id=tx_id, settlement=settlement)
