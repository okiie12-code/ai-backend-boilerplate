"""Request/response contracts for the AI endpoints."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=200_000)
    system: str | None = Field(default=None, max_length=50_000)
    max_tokens: int | None = Field(default=None, ge=1, le=128_000)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class StreamEvent(BaseModel):
    """One Server-Sent Event emitted by `/chat/stream`.

    Sequence: `start` -> `delta`* -> `done`, or `error` at any point after `start`.
    """

    type: Literal["start", "delta", "done", "error"]
    text: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    usage: Usage | None = None
    error: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200_000, description="Messy source text.")


class LeadExtraction(BaseModel):
    """Example extraction target: a sales lead pulled from an email, form, or chat log.

    Swap or add models like this one to build new extraction endpoints.
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = Field(description="Person's full name, if stated.")
    email: str | None = Field(description="Email address exactly as written, if present.")
    company: str | None = Field(description="Company or organisation name, if present.")
    budget_usd: int | None = Field(
        ge=0, description="Stated budget converted to whole US dollars, if any."
    )
    intent: Literal["purchase", "demo", "support", "partnership", "other"] = Field(
        description="Primary reason the person reached out."
    )
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(max_length=500, description="One or two sentence summary.")


class ExtractResponse(BaseModel):
    data: LeadExtraction
    model: str
    usage: Usage
