"""Application entry point: `uvicorn app.main:app`."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import ai, commands, health, tip, webhook
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logger import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.services.ai_service import build_client
from app.services.core_bot import MockCoreBot
from app.services.ledger import Ledger

log = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.use_json_logs)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.anthropic = build_client(settings)
        app.state.ledger = Ledger.demo()
        app.state.pending_plans = {}
        # Swap MockCoreBot for an HttpCoreBot(base_url=...) to forward to a real backend.
        app.state.core_bot = MockCoreBot()
        log.info("app.startup", environment=settings.environment, model=settings.llm_model)
        try:
            yield
        finally:
            await app.state.anthropic.close()
            log.info("app.shutdown")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        lifespan=lifespan,
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None,
    )

    # Middleware runs outermost-last-added: CORS wraps request context.
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(ai.router, prefix="/api/v1")
    app.include_router(commands.router, prefix="/api/v1")
    app.include_router(tip.router, prefix="/api/v1")
    app.include_router(webhook.router, prefix="/api/v1")
    return app


app = create_app()
