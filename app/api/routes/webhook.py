"""Chat-webhook adapter: the full 'messy text in, clean payload out' proxy flow.

    Discord/Kick webhook
        -> parse (schema-constrained model call)
        -> settle() deterministic validation   <-- the trust boundary
        -> forward clean SettlementRequest to the core bot
        -> return the core's receipt

This is the demo of the integration pattern. The core bot here is a mock
(`app.state.core_bot`); pointing it at a real backend is a one-line swap in the
app factory and changes nothing in this file.
"""

from fastapi import APIRouter

from app.api.deps import AIServiceDep, CoreBotDep
from app.core.exceptions import AppError
from app.core.logger import get_logger
from app.schemas.dwallet_intent import IntentRejected, ParsedIntent, settle
from app.schemas.webhook import ChatWebhookEvent, WebhookResult

router = APIRouter(prefix="/webhook", tags=["webhook"])
log = get_logger(__name__)

PARSE_INSTRUCTIONS = (
    "You convert a single chat message into a structured crypto tip/transfer intent. "
    "Extract only what the message states: the action, the numeric amount, whether it "
    "is fiat (dollars) vs the crypto asset, the uppercase ticker (Solana->SOL, etc.), "
    "and either a single recipient handle (without '@') or, for 'split among N', the "
    "split_count. Do not invent recipients, amounts or assets."
)


class IntentRejectedError(AppError):
    status_code = 422
    code = "intent_rejected"
    message = "The message could not be turned into a valid transaction."


@router.post("/chat", response_model=WebhookResult)
async def chat_webhook(
    event: ChatWebhookEvent, ai: AIServiceDep, core_bot: CoreBotDep
) -> WebhookResult:
    """Parse a chat message, validate it, and forward a clean payload to the core bot."""
    parsed: ParsedIntent
    parsed, _ = await ai.extract(event.text, ParsedIntent, instructions=PARSE_INSTRUCTIONS)

    try:
        settlement = settle(parsed)  # deterministic barrier; raw text stops here
    except IntentRejected as exc:
        log.info("webhook.rejected", platform=event.platform, problems=exc.problems)
        raise IntentRejectedError(details={"problems": exc.problems}) from exc

    receipt = await core_bot.submit(settlement)
    log.info("webhook.forwarded", platform=event.platform, tx_id=receipt.tx_id)
    return WebhookResult(
        platform=event.platform, parsed=parsed, forwarded=settlement, receipt=receipt
    )
