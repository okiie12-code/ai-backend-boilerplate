"""Natural-language-to-transaction demo endpoints (mock funds only).

Flow: POST /preview parses and validates a command and returns a confirmation
token; POST /execute applies that exact plan. Nothing moves without the second,
explicit call. GET /ledger shows the mock balances so you can watch them change.
"""

from fastapi import APIRouter

from app.api.deps import CommandServiceDep, LedgerDep
from app.schemas.commands import (
    CommandRequest,
    ExecuteRequest,
    PreviewResponse,
    ReceiptResponse,
)

router = APIRouter(prefix="/commands", tags=["commands"])


@router.get("/ledger")
async def ledger_snapshot(ledger: LedgerDep) -> dict[str, object]:
    """Current mock balances and activity scores."""
    return {
        "accounts": [
            {
                "username": a.username,
                "balances": {asset: str(amount) for asset, amount in a.balances.items()},
                "activity_score": a.activity_score,
            }
            for a in ledger.accounts()
        ]
    }


@router.post("/preview", response_model=PreviewResponse)
async def preview(body: CommandRequest, service: CommandServiceDep) -> PreviewResponse:
    """Parse a plain-English command into a validated plan. Moves no funds."""
    command = await service.parse(body.text)
    plan = service.resolve(command, sender=body.sender)
    feasible, reason = service.feasibility(plan)
    token = service.stage(plan) if feasible else None
    return PreviewResponse(
        command=command,
        plan=plan,
        feasible=feasible,
        reason=reason,
        confirmation_token=token,
    )


@router.post("/execute", response_model=ReceiptResponse)
async def execute(body: ExecuteRequest, service: CommandServiceDep) -> ReceiptResponse:
    """Apply the plan behind a confirmation token. Idempotent per token."""
    plan = await service.execute(body.confirmation_token)
    return ReceiptResponse(
        executed=True,
        sender=plan.sender,
        asset=plan.asset,
        transfers=plan.transfers,
        total=plan.total,
    )
