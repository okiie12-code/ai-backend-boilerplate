"""Contracts for the chat-webhook adapter demo.

Models an inbound message from a chat platform (Discord/Kick) and the result of
running it through the intent-parsing layer and forwarding it to a core bot.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.dwallet_intent import ParsedIntent, SettlementRequest

Platform = Literal["discord", "kick"]


class ChatWebhookEvent(BaseModel):
    """The shape a chat platform would POST to the adapter (trimmed to essentials)."""

    model_config = ConfigDict(extra="ignore")  # real webhooks carry many extra fields

    platform: Platform
    user_id: str = Field(min_length=1, max_length=64, description="Sender's platform user id.")
    text: str = Field(min_length=1, max_length=2000, description="Raw message text.")


class CoreBotReceipt(BaseModel):
    """What the (mock) core bot returns after accepting a clean settlement."""

    accepted: bool
    tx_id: str
    settlement: SettlementRequest


class WebhookResult(BaseModel):
    """Adapter response: what was parsed, what was forwarded, and the core's receipt."""

    platform: Platform
    parsed: ParsedIntent
    forwarded: SettlementRequest
    receipt: CoreBotReceipt
