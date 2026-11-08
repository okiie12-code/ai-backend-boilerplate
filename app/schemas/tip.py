"""Single-command tip extraction — the dwallet-style mapping.

A chat message like "give ten dollars of Solana to @crek" becomes a structured
payload of (action, amount, is_fiat_denominated, asset_ticker, target_user_id).

The model fills these fields **loosely** (strings, a raw number). The strict
rules — known ticker, valid handle, positive finite amount — are enforced by the
deterministic `verify` layer, not trusted from the model. See `app/services/tip.py`.
"""

from pydantic import BaseModel, ConfigDict, Field


class TipCommand(BaseModel):
    """Raw extraction target. Field *types* are permissive by design; validity is
    decided afterwards by the deterministic verifier."""

    model_config = ConfigDict(extra="forbid")

    action: str = Field(
        description="The verb of the request, lowercased, e.g. 'tip', 'send', 'airdrop', 'pay'."
    )
    amount: float = Field(
        description="The numeric amount. For 'ten dollars' this is 10; for '2.5 SOL' it is 2.5."
    )
    is_fiat_denominated: bool = Field(
        description=(
            "True if the amount is expressed in a fiat currency (dollars, USD, euros, etc.), "
            "so a price conversion is needed. False if it is expressed in the crypto asset itself."
        )
    )
    asset_ticker: str = Field(
        description=(
            "The crypto asset's ticker symbol in uppercase. Map names to tickers: "
            "Solana->SOL, Ethereum->ETH, Polygon/Matic->MATIC, Bitcoin->BTC. "
            "For a fiat 'ten dollars of Solana', the asset is still SOL."
        )
    )
    target_user_id: str = Field(
        description="The recipient's handle/username, without a leading '@'."
    )


class VerifiedTip(BaseModel):
    """A TipCommand after normalisation and deterministic validation."""

    action: str
    amount: float
    is_fiat_denominated: bool
    asset_ticker: str
    target_user_id: str
    # True when the amount is fiat and must be priced into `asset_ticker` before settlement.
    requires_price_conversion: bool


class TipRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000, description="Raw chat text.")


class TipResponse(BaseModel):
    raw: TipCommand
    verified: VerifiedTip
    model: str
