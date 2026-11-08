"""dwallet-style tip command extraction.

POST a raw chat message; get back the strictly-typed payload the bot would act
on — after deterministic verification. Extraction uses the model configured in
settings (`LLM_MODEL`); the route never hardcodes a model string.
"""

from fastapi import APIRouter

from app.api.deps import AIServiceDep
from app.schemas.tip import TipCommand, TipRequest, TipResponse
from app.services.tip import verify

router = APIRouter(prefix="/tip", tags=["tip"])

EXTRACT_INSTRUCTIONS = (
    "You extract a single crypto tipping/payment command from a chat message. "
    "Identify the action verb, the numeric amount, whether the amount is in fiat "
    "(dollars/USD/euros) rather than the crypto asset, the asset's uppercase ticker "
    "(map coin names to tickers, e.g. Solana->SOL), and the recipient's handle without '@'. "
    "Use only what the message states; do not invent a recipient, amount or asset."
)


@router.post("/extract", response_model=TipResponse)
async def extract_tip(body: TipRequest, service: AIServiceDep) -> TipResponse:
    """Extract and verify a tip command from raw chat text.

    Returns `422 tip_invalid` with a list of problems if the payload fails
    deterministic validation (unknown ticker, bad handle, non-positive amount).
    """
    raw, message = await service.extract(body.text, TipCommand, instructions=EXTRACT_INSTRUCTIONS)
    verified = verify(raw)  # raises TipValidationError on bad payloads
    return TipResponse(raw=raw, verified=verified, model=message.model)
