"""FastAPI dependencies. Override these in tests via `app.dependency_overrides`."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.ai_service import AIService


def get_ai_service(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> AIService:
    return AIService(request.app.state.anthropic, settings)


SettingsDep = Annotated[Settings, Depends(get_settings)]
AIServiceDep = Annotated[AIService, Depends(get_ai_service)]
