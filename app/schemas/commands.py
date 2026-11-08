"""Contracts for the natural-language-to-transaction demo.

`ParsedCommand` is the *only* thing the model produces — a proposal. It never
names amounts the app will move without the deterministic layer re-deriving and
re-validating them, and it never executes anything.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Action = Literal["tip", "airdrop", "send"]
Asset = Literal["MATIC", "USDC", "ETH"]
Distribution = Literal["each", "split_total"]


class ParsedCommand(BaseModel):
    """What the LLM extracts from a free-text command. Structure only, no execution."""

    model_config = ConfigDict(extra="forbid")

    action: Action = Field(description="Kind of transfer requested.")
    asset: Asset = Field(description="Token/currency symbol.")
    amount: float = Field(
        gt=0,
        description=(
            "The numeric amount in the command. If the command gives a per-person "
            "amount use distribution='each'; if it gives one total to divide use "
            "distribution='split_total'."
        ),
    )
    distribution: Distribution = Field(
        description="'each' = amount per recipient; 'split_total' = amount split across recipients."
    )
    recipients: list[str] = Field(
        default_factory=list,
        description="Explicit usernames named in the command (without @). Empty if none named.",
    )
    recipient_selector: str | None = Field(
        default=None,
        description="How to pick recipients when not named, e.g. 'most active users'.",
    )
    recipient_count: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="How many recipients the selector should resolve to, if a count is given.",
    )
    note: str | None = Field(
        default=None, max_length=200, description="Any memo/message, if present."
    )


# -- request/response models ---------------------------------------------


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000, description="The plain-English command.")
    sender: str = Field(default="treasury", description="Account the funds come from (mock).")


class TransferLine(BaseModel):
    recipient: str
    amount: Decimal


class Plan(BaseModel):
    """The deterministic, fully-resolved transaction derived from a ParsedCommand."""

    sender: str
    asset: Asset
    transfers: list[TransferLine]
    total: Decimal


class PreviewResponse(BaseModel):
    command: ParsedCommand
    plan: Plan
    feasible: bool
    reason: str | None = None
    # Present only when feasible — hand it back to /execute to apply exactly this plan.
    confirmation_token: str | None = None


class ReceiptResponse(BaseModel):
    executed: bool
    sender: str
    asset: Asset
    transfers: list[TransferLine]
    total: Decimal


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmation_token: str = Field(description="Token from a prior /preview call.")
