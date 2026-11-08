"""Turn a plain-English command into a safe, validated transaction.

The pipeline, and where trust sits at each step:

    text ──LLM──▶ ParsedCommand ──deterministic──▶ Plan ──preview──▶ token
                                                                        │
                                                      /execute ──ledger.apply (atomic)

Only the first arrow involves the model, and it only ever produces a *proposal*.
Everything that decides who gets paid, how much, and whether the sender can
afford it is deterministic Python. Funds move only after an explicit second call
(`execute`) carrying a token for one specific, already-validated plan.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from app.core.exceptions import AppError
from app.schemas.commands import ParsedCommand, Plan, TransferLine
from app.services.ai_service import AIService
from app.services.ledger import InsufficientFunds, Ledger, UnknownAccount, quantize

PARSE_INSTRUCTIONS = (
    "You convert a plain-English payment command into a structured proposal. "
    "Extract only what the text states; never invent usernames, amounts or assets. "
    "Use distribution='each' when the amount is per recipient (e.g. 'tip 2 USDC to alice, bob'), "
    "and distribution='split_total' when one total is divided among recipients "
    "(e.g. 'airdrop 5 MATIC to the 3 most active users'). "
    "Put explicitly named usernames in recipients; describe any group in recipient_selector "
    "with recipient_count. You are not executing anything."
)


class CommandError(AppError):
    status_code = 422
    code = "command_invalid"
    message = "The command could not be turned into a valid transaction."


class UnknownToken(AppError):
    status_code = 404
    code = "unknown_token"
    message = "No pending transaction matches that confirmation token."


class PlanNoLongerFeasible(AppError):
    status_code = 409
    code = "plan_infeasible"
    message = "The transaction is no longer feasible against current balances."


class CommandService:
    def __init__(self, ledger: Ledger, ai: AIService, pending: dict[str, Plan]) -> None:
        self._ledger = ledger
        self._ai = ai
        # token -> plan, shared across requests (lives on app.state). In a real
        # service this would be a short-TTL store such as Redis.
        self._pending = pending

    async def parse(self, text: str) -> ParsedCommand:
        command, _ = await self._ai.extract(text, ParsedCommand, instructions=PARSE_INSTRUCTIONS)
        return command

    def resolve(self, command: ParsedCommand, *, sender: str) -> Plan:
        """Derive the concrete transfers. Deterministic — no model involved."""
        if not self._ledger.exists(sender):
            raise CommandError(f"Unknown sender account '{sender}'.")

        recipients = self._resolve_recipients(command, sender=sender)
        if not recipients:
            raise CommandError("The command does not identify any recipients.")

        amount = Decimal(str(command.amount))
        if command.distribution == "each":
            lines = [TransferLine(recipient=r, amount=quantize(amount)) for r in recipients]
        else:  # split_total — divide, giving any rounding remainder to the first recipient
            share = quantize(amount / Decimal(len(recipients)))
            lines = [TransferLine(recipient=r, amount=share) for r in recipients]
            remainder = quantize(amount - share * Decimal(len(recipients)))
            if remainder:
                lines[0] = TransferLine(
                    recipient=lines[0].recipient, amount=quantize(lines[0].amount + remainder)
                )

        total = quantize(sum((line.amount for line in lines), Decimal(0)))
        if total <= 0:
            raise CommandError("Resolved transaction total is zero.")
        return Plan(sender=sender, asset=command.asset, transfers=lines, total=total)

    def _resolve_recipients(self, command: ParsedCommand, *, sender: str) -> list[str]:
        if command.recipients:
            unknown = [r for r in command.recipients if not self._ledger.exists(r)]
            if unknown:
                raise CommandError(f"Unknown recipient account(s): {', '.join(unknown)}.")
            # de-dupe, preserve order, never pay the sender
            seen: dict[str, None] = {}
            for r in command.recipients:
                if r != sender:
                    seen.setdefault(r, None)
            return list(seen)
        if command.recipient_selector and command.recipient_count:
            return self._ledger.top_active(command.recipient_count, exclude=sender)
        raise CommandError("The command does not identify any recipients.")

    def feasibility(self, plan: Plan) -> tuple[bool, str | None]:
        available = self._ledger.get(plan.sender).balance(plan.asset)
        if plan.total > available:
            return False, f"{plan.sender} has {available} {plan.asset}, needs {plan.total}."
        return True, None

    def stage(self, plan: Plan) -> str:
        token = uuid.uuid4().hex
        self._pending[token] = plan
        return token

    async def execute(self, token: str) -> Plan:
        plan = self._pending.get(token)
        if plan is None:
            raise UnknownToken()
        try:
            await self._ledger.apply(
                sender=plan.sender,
                asset=plan.asset,
                transfers=[(line.recipient, line.amount) for line in plan.transfers],
                idempotency_key=token,  # replay-safe: re-executing this token moves nothing
            )
        except InsufficientFunds as exc:
            raise PlanNoLongerFeasible(str(exc)) from exc
        except UnknownAccount as exc:
            raise CommandError(f"Account disappeared before execution: {exc}") from exc
        return plan
