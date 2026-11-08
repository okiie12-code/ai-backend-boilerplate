"""FastAPI dependencies. Override these in tests via `app.dependency_overrides`."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.schemas.commands import Plan
from app.services.ai_service import AIService
from app.services.command_service import CommandService
from app.services.ledger import Ledger


def get_ai_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AIService:
    return AIService(request.app.state.anthropic, settings)


def get_ledger(request: Request) -> Ledger:
    ledger: Ledger = request.app.state.ledger
    return ledger


def get_command_service(
    request: Request,
    ledger: Annotated[Ledger, Depends(get_ledger)],
    ai: Annotated[AIService, Depends(get_ai_service)],
) -> CommandService:
    pending: dict[str, Plan] = request.app.state.pending_plans
    return CommandService(ledger, ai, pending)


SettingsDep = Annotated[Settings, Depends(get_settings)]
AIServiceDep = Annotated[AIService, Depends(get_ai_service)]
LedgerDep = Annotated[Ledger, Depends(get_ledger)]
CommandServiceDep = Annotated[CommandService, Depends(get_command_service)]
