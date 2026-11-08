"""Application error types and the handlers that turn them into consistent JSON.

Every error response has the same shape:

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": ...}}
"""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logger import get_logger

log = get_logger(__name__)


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: Any = None,
        retry_after: int | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details
        self.retry_after = retry_after
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            body["details"] = self.details
        if self.retry_after is not None:
            body["retry_after"] = self.retry_after
        return body


class LLMRateLimitError(AppError):
    status_code = 429
    code = "llm_rate_limited"
    message = "The AI provider is rate limiting requests. Retry shortly."


class LLMTimeoutError(AppError):
    status_code = 504
    code = "llm_timeout"
    message = "The AI provider did not respond in time."


class LLMUnavailableError(AppError):
    status_code = 503
    code = "llm_unavailable"
    message = "The AI provider is temporarily unavailable."


class LLMBadRequestError(AppError):
    status_code = 400
    code = "llm_bad_request"
    message = "The AI provider rejected the request."


class LLMConfigurationError(AppError):
    status_code = 500
    code = "llm_misconfigured"
    message = "The AI provider is not configured correctly."


class LLMRefusalError(AppError):
    status_code = 422
    code = "llm_refused"
    message = "The model declined to complete this request."


class LLMBadResponseError(AppError):
    status_code = 502
    code = "llm_bad_response"
    message = "The model returned output that did not match the expected format."


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    request: Request,
    status_code: int,
    body: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {**body, "request_id": _request_id(request)}
    all_headers = dict(headers or {})
    if (rid := _request_id(request)) is not None:
        all_headers["X-Request-ID"] = rid
    return JSONResponse(
        status_code=status_code,
        content={"error": jsonable_encoder(body)},
        headers=all_headers,
    )


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    log.warning("app_error", code=exc.code, status_code=exc.status_code, error=exc.message)
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return _error_response(request, exc.status_code, exc.to_dict(), headers)


async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return _error_response(
        request,
        422,
        {"code": "validation_error", "message": "Invalid request.", "details": exc.errors()},
    )


async def _http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    return _error_response(
        request,
        exc.status_code,
        {"code": f"http_{exc.status_code}", "message": str(exc.detail)},
        dict(exc.headers or {}),
    )


async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", exc_info=exc)
    return _error_response(request, 500, AppError().to_dict())


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_exception_handler(Exception, _unhandled_error_handler)
