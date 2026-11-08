"""AI endpoints: SSE chat streaming and structured extraction."""

from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import AIServiceDep
from app.core.exceptions import AppError
from app.core.logger import get_logger
from app.schemas.ai import (
    ChatRequest,
    ExtractRequest,
    ExtractResponse,
    LeadExtraction,
    StreamEvent,
    Usage,
)

router = APIRouter(prefix="/ai", tags=["ai"])
log = get_logger(__name__)

LEAD_INSTRUCTIONS = (
    "You extract sales-lead information from messy inbound text such as emails, "
    "web-form submissions and chat transcripts. Use only facts stated in the text. "
    "Use null for any field the text does not state; do not guess names, emails or budgets."
)


def _sse(event: StreamEvent) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_stream(body: ChatRequest, service: AIServiceDep) -> StreamingResponse:
    """Stream a Claude response as Server-Sent Events (`start`, `delta`, `done`, `error`)."""
    events = service.stream_chat(body.prompt, system=body.system, max_tokens=body.max_tokens)

    # Pull the first event before committing to a 200, so connection-time
    # failures (429, auth, bad request) come back as normal JSON errors.
    try:
        first = await anext(events)
    except BaseException:
        await events.aclose()
        raise

    async def body_iter() -> AsyncIterator[str]:
        try:
            yield _sse(first)
            async for event in events:
                yield _sse(event)
        except AppError as exc:
            log.warning("llm.stream.error", code=exc.code, error=exc.message)
            yield _sse(StreamEvent(type="error", error=exc.to_dict()))
        finally:
            await events.aclose()

    return StreamingResponse(
        body_iter(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/extract/lead", response_model=ExtractResponse)
async def extract_lead(body: ExtractRequest, service: AIServiceDep) -> ExtractResponse:
    """Turn messy inbound text into a validated `LeadExtraction` record."""
    data, message = await service.extract(body.text, LeadExtraction, instructions=LEAD_INSTRUCTIONS)
    return ExtractResponse(
        data=data,
        model=message.model,
        usage=Usage(
            input_tokens=message.usage.input_tokens, output_tokens=message.usage.output_tokens
        ),
    )
