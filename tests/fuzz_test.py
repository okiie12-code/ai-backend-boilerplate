"""Concurrency / double-spend correctness harness for THIS boilerplate.

What it does
------------
Fires many simultaneous requests at the running app and checks that the ledger's
invariants hold under contention — i.e. that the endpoint serialises conflicting
writes (the in-memory equivalent of row-level locking) and never double-applies a
transaction. It is a QA tool for verifying your own service, not a general probe.

Two scenarios:
  1. Idempotency race — fire N concurrent /commands/execute calls for the SAME
     confirmation token. A correct backend applies the transfer exactly once;
     a broken one pays the recipient N times (the double-spend bug).
  2. Competing-spend race — stage two plans that each need most of one account's
     balance, then execute both at once. Exactly one must succeed.

Scope / safety
--------------
This targets a service you run and own. It refuses any non-loopback host: it is
not an audit tool for third-party systems, and running concurrent transaction
floods against someone else's production service is not authorised by this file.
Point it only at your own local instance (or one you have written permission to
load-test), started with a valid ANTHROPIC_API_KEY so /commands/preview works.

Usage
-----
    uvicorn app.main:app --port 8000          # in one shell, with a real key
    python tests/fuzz_test.py                  # in another
    python tests/fuzz_test.py --n 50 --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import time
from decimal import Decimal
from urllib.parse import urlparse

import httpx

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _assert_loopback(base_url: str) -> None:
    host = urlparse(base_url).hostname or ""
    if host not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"Refusing to run against non-loopback host '{host}'.\n"
            "This harness tests a service you run locally, not a third-party system.\n"
            "Point it at 127.0.0.1 (your own instance)."
        )


# -- microsecond logging --------------------------------------------------

_T0 = time.perf_counter_ns()


def log(tag: str, msg: str) -> None:
    elapsed_us = (time.perf_counter_ns() - _T0) / 1_000
    print(f"[{elapsed_us:12.3f}µs] {tag:<9} {msg}", flush=True)


# -- helpers --------------------------------------------------------------


async def preview(client: httpx.AsyncClient, text: str, sender: str) -> dict[str, object]:
    r = await client.post("/api/v1/commands/preview", json={"text": text, "sender": sender})
    r.raise_for_status()
    return r.json()


async def balance(client: httpx.AsyncClient, user: str, asset: str = "MATIC") -> Decimal:
    r = await client.get("/api/v1/commands/ledger")
    r.raise_for_status()
    account = next(a for a in r.json()["accounts"] if a["username"] == user)
    return Decimal(account["balances"].get(asset, "0"))


async def timed_execute(client: httpx.AsyncClient, token: str, i: int) -> tuple[int, float]:
    start = time.perf_counter_ns()
    log("FIRE", f"req #{i:02d} → execute token …{token[-6:]}")
    r = await client.post("/api/v1/commands/execute", json={"confirmation_token": token})
    dt_us = (time.perf_counter_ns() - start) / 1_000
    log("RECV", f"req #{i:02d} ← HTTP {r.status_code} in {dt_us:9.3f}µs")
    return r.status_code, dt_us


# -- scenario 1: idempotency / double-apply race --------------------------


async def scenario_idempotency(client: httpx.AsyncClient, n: int) -> bool:
    log("SETUP", "scenario 1: N concurrent executes of ONE token")
    plan = await preview(client, "airdrop 6 MATIC to top 3", "treasury")
    token = plan["confirmation_token"]
    if not token:
        raise SystemExit(f"preview was not feasible: {plan.get('reason')}")
    total = Decimal(str(plan["plan"]["total"]))  # type: ignore[index]

    before = await balance(client, "treasury")
    log("ARM", f"token …{str(token)[-6:]} staged; treasury={before} MATIC; firing {n} at once")

    results = await asyncio.gather(*(timed_execute(client, str(token), i) for i in range(n)))
    after = await balance(client, "treasury")

    ok_200 = sum(1 for code, _ in results if code == 200)
    spent = before - after
    applied_times = (spent / total) if total else Decimal(0)
    log("LEDGER", f"treasury {before} → {after}  (spent {spent}, = {applied_times}x the plan)")

    passed = spent == total
    verdict = "PASS — applied exactly once" if passed else f"FAIL — DOUBLE-SPEND (x{applied_times})"
    log("RESULT", f"{verdict}; {ok_200}/{n} returned 200 (idempotent replays also 200)")
    return passed


# -- scenario 2: competing spends -----------------------------------------


async def scenario_competing(client: httpx.AsyncClient) -> bool:
    log("SETUP", "scenario 2: two plans competing for one account's funds")
    # dave starts with 10 MATIC; two 8-MATIC sends cannot both clear.
    a = await preview(client, "send 8 MATIC to alice", "dave")
    b = await preview(client, "send 8 MATIC to bob", "dave")
    tokens = [a.get("confirmation_token"), b.get("confirmation_token")]
    if not all(tokens):
        log("SKIP", "dave not funded as expected; skipping (run against a fresh instance)")
        return True

    before = await balance(client, "dave")
    results = await asyncio.gather(
        *(timed_execute(client, str(t), i) for i, t in enumerate(tokens))
    )
    after = await balance(client, "dave")

    wins = sum(1 for code, _ in results if code == 200)
    passed = wins == 1 and after == before - Decimal("8")
    log("LEDGER", f"dave {before} → {after}; winners={wins}")
    log(
        "RESULT",
        "PASS — exactly one spend cleared" if passed else "FAIL — overdraft or both cleared",
    )
    return passed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=50, help="concurrent requests in scenario 1")
    args = parser.parse_args()
    _assert_loopback(args.url)

    log("START", f"target={args.url}  concurrency={args.n}")
    async with httpx.AsyncClient(base_url=args.url, timeout=30.0) as client:
        try:
            await client.get("/health")
        except httpx.ConnectError:
            raise SystemExit(
                f"No server at {args.url}. Start: uvicorn app.main:app --port 8000"
            ) from None

        s1 = await scenario_idempotency(client, args.n)
        print()
        s2 = await scenario_competing(client)

    print()
    log("DONE", f"scenario1={'PASS' if s1 else 'FAIL'}  scenario2={'PASS' if s2 else 'FAIL'}")
    return 0 if (s1 and s2) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
