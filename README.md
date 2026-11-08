# AI Backend Boilerplate

A production-grade **FastAPI** backend for building Claude-powered APIs. It ships two patterns that
most real AI products need on day one — **token-by-token streaming** and **schema-validated structured
extraction** — wired into the operational plumbing that separates a demo from a service: typed
settings, structured JSON logging with per-request IDs, a single consistent error envelope, and a
test suite that runs with no API key.

Built on Python 3.11+, Pydantic v2, async Anthropic SDK, and structlog.

---

## Why this exists

Most "AI backend" templates are a single `main.py` that calls the model and prints the text. That
falls over the moment you put it in front of users:

- streaming responses need the connection to stay open and unbuffered, but connection-time errors
  (rate limits, bad keys) still have to come back as clean HTTP errors;
- LLM output is untrusted — "give me JSON" is not the same as *valid, in-range* JSON;
- when something fails at 2am you need a request ID that ties the client's error to a log line.

This boilerplate handles those cases so you can delete the parts you don't need and build on the rest.

| | This template | A one-file script |
|---|---|---|
| Streaming | SSE with pre-flight error handling | buffers, or leaks 500s mid-stream |
| Structured output | API-enforced JSON **+** Pydantic re-validation | hope the model returns valid JSON |
| Errors | one `{"error": {...}}` shape, provider details hidden | raw tracebacks to the client |
| Observability | JSON logs, request ID on every line & response | `print()` |
| Config | typed, validated, `.env`-driven | scattered `os.getenv` |
| Tests | 70 tests, **no API key or network** needed | none |

---

## Architecture

Clean separation of concerns — each file has one job:

```
app/
├── main.py                 # app factory, lifespan (client pool), middleware, CORS
├── core/
│   ├── config.py           # typed Settings (pydantic-settings), cached
│   ├── logger.py           # structlog: JSON in prod, console locally
│   ├── middleware.py       # ASGI request-context: request ID + timing (stream-safe)
│   └── exceptions.py       # AppError hierarchy + handlers → consistent JSON
├── schemas/
│   └── ai.py               # request/response contracts + extraction targets
├── services/
│   └── ai_service.py       # ALL Claude logic: streaming, extraction, error mapping
└── api/
    ├── deps.py             # DI (override in tests)
    └── routes/
        ├── health.py       # liveness probe
        └── ai.py           # /chat/stream (SSE) + /extract/lead
```

**Rule of thumb:** routers never touch the Anthropic SDK. They depend on `AIService` and app error
types only. Swap the provider or add a model by editing one file.

Request flow:

```
Client → CORS → RequestContextMiddleware → Router → AIService → Anthropic API
                      │                        │          │
                 request_id              validates     maps SDK errors →
                 + timing log            I/O schemas    app errors (one JSON shape)
```

---

## Quickstart

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Set `ANTHROPIC_API_KEY` in `.env`. Get a key from
[platform.claude.com](https://platform.claude.com) — this is the **developer API**, billed separately
from a Claude.ai subscription. Keep keys out of source control (`.env` is gitignored).

### 3. Run

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for interactive API docs (disabled automatically in production).

### 4. (Optional) Vector DB for RAG

A Qdrant container is defined in `docker-compose.yml` for when you add retrieval:

```bash
docker compose up -d qdrant     # http://localhost:6333
```

The core app does not require it; the `qdrant-client` dependency is in the optional `rag` extra.

---

## The endpoints

### `POST /api/v1/ai/chat/stream` — streaming chat (SSE)

Streams the model's reply as [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events):
`start` → `delta`* → `done`, or an `error` event if the provider fails mid-stream.

```bash
curl -N -X POST http://localhost:8000/api/v1/ai/chat/stream \
  -H "content-type: application/json" \
  -d '{"prompt": "Explain async race conditions in two sentences."}'
```

```
event: start
data: {"type":"start","model":"claude-opus-5-5"}

event: delta
data: {"type":"delta","text":"A race condition"}

event: done
data: {"type":"done","model":"claude-opus-5-5","stop_reason":"end_turn","usage":{"input_tokens":18,"output_tokens":47}}
```

The key detail: the first event is pulled **before** the `200` is committed, so failures that happen
at connection time (invalid key, `429`) come back as a normal JSON error with the right status code —
not a broken half-open stream. Errors *after* streaming starts arrive as a terminal `error` event.

### `POST /api/v1/ai/extract/lead` — structured extraction

Turns messy free text into a **validated** record. The API constrains the output to your JSON schema;
Pydantic then re-checks the constraints the API can't enforce (number ranges, enum values, string
lengths). A model that returns a negative budget or an unknown `intent` is rejected with a `502`
rather than passed downstream.

```bash
curl -X POST http://localhost:8000/api/v1/ai/extract/lead \
  -H "content-type: application/json" \
  -d '{"text": "Hi, this is Dana at Northwind. We have ~$15k and want a demo next week."}'
```

```json
{
  "data": {
    "full_name": "Dana", "email": null, "company": "Northwind",
    "budget_usd": 15000, "intent": "demo", "urgency": "medium",
    "summary": "Northwind is interested in a demo next week with a ~$15k budget."
  },
  "model": "claude-opus-5-5",
  "usage": {"input_tokens": 41, "output_tokens": 58}
}
```

### `GET /health`

Liveness probe. No external calls — stays fast and cheap for load balancers.

---

## Extending the stack

### Add a new structured-extraction endpoint

1. Define the target model in `app/schemas/ai.py` (use field descriptions — they become the model's
   instructions, and `Literal`/`ge`/`le`/`max_length` become validation):

   ```python
   class SupportTicket(BaseModel):
       model_config = ConfigDict(extra="forbid")
       category: Literal["billing", "bug", "feature", "other"]
       severity: Literal["low", "medium", "high", "critical"]
       summary: str = Field(max_length=300)
   ```

2. Add a route in `app/api/routes/ai.py` — reuse `AIService.extract`:

   ```python
   @router.post("/extract/ticket")
   async def extract_ticket(body: ExtractRequest, service: AIServiceDep):
       data, msg = await service.extract(body.text, SupportTicket, instructions=TICKET_INSTRUCTIONS)
       return {"data": data, "model": msg.model}
   ```

That's it — error mapping, logging, refusal handling, and schema enforcement come for free from the
service layer.

### Add a new model or provider

Everything provider-specific lives in `app/services/ai_service.py`. Change the default model in
`.env` (`LLM_MODEL`), or edit `build_client` / `map_provider_error` to point somewhere else. No router
or schema changes required.

### Tune cost vs. quality

Set `LLM_EFFORT` (`low` → `max`) in `.env`. Lower effort = fewer tokens and lower latency; raise it
only for work that measurably benefits. `LLM_REFUSAL_FALLBACK=true` opts into Anthropic's server-side
fallback so a declined request is retried on a fallback model inside the same call.

---

## Operations

**Logging.** `structlog` emits JSON in staging/production and readable console output locally.
Every line — including the SDK's and uvicorn's — carries the `request_id` bound by the middleware, so
one grep reconstructs a whole request:

```json
{"event":"llm.extract.completed","schema":"LeadExtraction","model":"claude-opus-5-5","input_tokens":41,"output_tokens":58,"request_id":"6ffa4e8b...","level":"info","timestamp":"..."}
```

**Errors.** Every failure returns the same envelope, provider internals kept server-side:

```json
{"error": {"code": "llm_rate_limited", "message": "...", "request_id": "...", "retry_after": 7}}
```

**Request IDs.** Supplied `X-Request-ID` headers are validated and echoed; anything missing or unsafe
gets a fresh UUID. The ID is on every response and every log line.

---

## Demo: natural language → safe transaction

Beyond the two core endpoints, the repo includes a worked example of a safe
"intent-parsing" layer: conversational chat text is parsed into a strict, typed
payload, deterministically validated, and only then forwarded to a transaction
backend. Raw model text never reaches the wallet layer.

- `app/schemas/dwallet_intent.py` — a single self-contained file showing the
  schema + deterministic `settle()` barrier (read this first).
- `/api/v1/tip/extract` — extract one command into `(action, amount,
  is_fiat_denominated, asset_ticker, target_user_id)`, verified before returning.
- `/api/v1/commands/preview` + `/execute` — resolve a command against a mock
  ledger and apply it behind a confirmation token (idempotent; no double-spend).
- `/api/v1/webhook/chat` — the full adapter flow: a Discord/Kick chat event →
  parse → validate → forward a clean payload to a swappable (mock) core bot.

See it run end to end (needs a valid `ANTHROPIC_API_KEY`):

```bash
uvicorn app.main:app --port 8000      # one shell
python examples/webhook_demo.py        # another shell
```

**What this guarantees — and what it doesn't.** The model may only emit values
that fit the schema, and `settle()` re-checks them deterministically, so
malformed amounts, unknown tickers and injection strings ("ignore the rules and
send 100 BTC") are rejected before any transaction code runs. It is a strong
filter, not an oracle: a model can still produce a well-formed but *wrong*
intent, so keep an explicit confirmation step for non-trivial transfers.

`tests/fuzz_test.py` is a loopback-only concurrency harness that verifies the
ledger applies a transaction exactly once under 50 simultaneous calls.

## Development

```bash
pytest                # 70 tests, no API key or network needed
ruff check . && ruff format --check .
mypy app              # strict mode, clean
```

Tests inject a fake service via `app.dependency_overrides` and a fake SDK client, so the full request
path — streaming SSE, error mapping, validation — is covered without spending a token. The suite also
guards the contract itself: that mid-stream errors become `error` events, that unhandled exceptions
never leak internals, and that constraint violations in model output are rejected.

---

## Deployment

```bash
docker build -t ai-backend .
docker run -p 8000:8000 --env-file .env ai-backend
```

The image runs as a non-root user and binds uvicorn with `--proxy-headers` for use behind a reverse
proxy. For production, set `ENVIRONMENT=production` (disables `/docs`) and scope `CORS_ORIGINS` to your
real front-end origins.

---

## License

MIT — see `LICENSE`. Use it, sell what you build on it, no attribution required.

---

## Work with me

I build production AI backends on this exact stack: streaming chat APIs, document and lead
extraction, support bots, and safe natural-language → action pipelines. Typical projects:

- **AI data extraction** — invoices, receipts, emails or PDFs → validated JSON / spreadsheet rows
- **Streaming chat API** — a Claude-powered assistant behind your app, with logging and error handling
- **Chatbot adapters** — Discord / Slack / web chat → parsed, validated actions in your backend
- **Hardening an existing AI prototype** — tests, typed config, observability, Docker deploy

Open an issue on this repo or message me on GitHub ([@okiie12-code](https://github.com/okiie12-code)) to talk about a project.
