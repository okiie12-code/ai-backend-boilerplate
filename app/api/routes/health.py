from typing import Any

from fastapi import APIRouter

from app import __version__
from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: SettingsDep) -> dict[str, Any]:
    """Liveness probe. Makes no external calls, so it stays fast and cheap."""
    return {
        "status": "ok",
        "version": __version__,
        "environment": settings.environment,
        "model": settings.llm_model,
    }
