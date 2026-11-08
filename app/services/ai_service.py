"""Claude integration: streaming chat and schema-validated structured extraction.

All provider-specific details (SDK calls, error types, refusal handling) live
here. Routers depend only on the `AIService` interface and app error types.
"""

from collections.abc import AsyncGenerator, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import anthropic
from anthropic.types import OutputConfigParam
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.exceptions import (
    AppError,
    LLMBadRequestError,
    LLMBadResponseError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.core.logger import get_logger
from app.schemas.ai import StreamEvent, Usage

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_client(settings: Settings) -> anthropic.AsyncAnthropic:
    """Create the shared async client. One instance per process; it pools connections."""
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    return anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,  # SDK retries 408/409/429/5xx with backoff
    )


def _retry_after(exc: anthropic.APIStatusError) -> int | None:
    value = exc.response.headers.get("retry-after")
    try:
        return max(0, int(float(value))) if value is not None else None
    except ValueError:
        return None


def map_provider_error(exc: Exception) -> AppError:
    """Translate SDK exceptions into app errors. Most specific classes first."""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError()
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError(retry_after=_retry_after(exc))
    if isinstance(
        exc,
        anthropic.AuthenticationError | anthropic.PermissionDeniedError | anthropic.NotFoundError,
    ):
        # Details stay in server logs; clients get a generic message.
        log.error("llm.configuration_error", error=str(exc), status_code=exc.status_code)
        return LLMConfigurationError()
    if isinstance(exc, anthropic.BadRequestError):
        return LLMBadRequestError(details={"provider_message": exc.message})
    if isinstance(exc, anthropic.APIStatusError):
        log.warning("llm.provider_error", status_code=exc.status_code, error=str(exc))
        return LLMUnavailableError(retry_after=_retry_after(exc))
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMUnavailableError()
    raise exc


def _is_missing_credentials(exc: TypeError) -> bool:
    # The SDK raises a bare TypeError at request-build time when no api_key /
    # auth_token / credentials can be resolved -- before any HTTP call, so it is
    # not an AnthropicError. Map it to a configuration error instead of a 500.
    return "authentication method" in str(exc).lower()


@contextmanager
def provider_errors() -> Iterator[None]:
    try:
        yield
    except anthropic.AnthropicError as exc:
        raise map_provider_error(exc) from exc
    except TypeError as exc:
        if _is_missing_credentials(exc):
            log.error("llm.configuration_error", error="no API credentials resolved")
            raise LLMConfigurationError() from exc
        raise


class AIService:
    def __init__(self, client: anthropic.AsyncAnthropic, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    # -- request building -------------------------------------------------

    def _common_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self._settings.llm_model}
        if self._settings.llm_refusal_fallback:
            kwargs["extra_headers"] = {"anthropic-beta": _FALLBACK_BETA}
            kwargs["extra_body"] = {"fallbacks": "default"}
        return kwargs

    def _output_config(self, **extra: Any) -> OutputConfigParam:
        return cast(OutputConfigParam, {"effort": self._settings.llm_effort, **extra})

    # -- streaming chat ---------------------------------------------------

    async def stream_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield `start`, then text `delta`s, then `done`.

        Errors raised before `start` (auth, rate limit, bad request) surface as
        normal HTTP errors; errors after it are raised from the iterator and the
        router turns them into an SSE `error` event.
        """
        kwargs = self._common_kwargs()
        if system:
            kwargs["system"] = system

        with provider_errors():
            async with self._client.messages.stream(
                max_tokens=max_tokens or self._settings.llm_stream_max_tokens,
                output_config=self._output_config(),
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            ) as stream:
                yield StreamEvent(type="start", model=self._settings.llm_model)
                async for text in stream.text_stream:
                    yield StreamEvent(type="delta", text=text)
                final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            raise LLMRefusalError(details=_refusal_details(final))

        log.info(
            "llm.stream.completed",
            model=final.model,
            stop_reason=final.stop_reason,
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )
        yield StreamEvent(
            type="done",
            model=final.model,
            stop_reason=final.stop_reason,
            usage=Usage(
                input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens
            ),
        )

    # -- structured extraction -------------------------------------------

    async def extract(
        self,
        text: str,
        schema: type[T],
        *,
        instructions: str,
    ) -> tuple[T, anthropic.types.Message]:
        """Extract `schema` from `text` using constrained JSON output, then validate it.

        The API enforces the JSON shape; constraints the API cannot enforce
        (ranges, lengths) are moved into field descriptions by the SDK and
        re-checked here by Pydantic.
        """
        output_format = {
            "type": "json_schema",
            "schema": anthropic.transform_schema(schema),
        }
        with provider_errors():
            message = await self._client.messages.create(
                max_tokens=self._settings.llm_max_tokens,
                system=instructions,
                output_config=self._output_config(format=output_format),
                messages=[{"role": "user", "content": text}],
                **self._common_kwargs(),
            )

        if message.stop_reason == "refusal":
            raise LLMRefusalError(details=_refusal_details(message))
        if message.stop_reason == "max_tokens":
            raise LLMBadResponseError("The model ran out of output tokens before finishing.")

        raw = next((b.text for b in message.content if b.type == "text"), None)
        if raw is None:
            raise LLMBadResponseError("The model returned no text content.")
        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as exc:
            log.warning("llm.extract.validation_failed", errors=exc.errors(include_input=False))
            raise LLMBadResponseError(
                details=exc.errors(include_input=False, include_url=False)
            ) from exc

        log.info(
            "llm.extract.completed",
            schema=schema.__name__,
            model=message.model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return parsed, message


def _refusal_details(message: anthropic.types.Message) -> dict[str, Any] | None:
    details = getattr(message, "stop_details", None)
    if details is None:
        return None
    return {"category": getattr(details, "category", None)}
