# DiaLens API

FastAPI service for diabetes risk screening with TensorFlow inference,
feature scaling, explainability, and Indonesian health recommendations also list of hospitals.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed.

## Environment Setup

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` if you want
LLM-generated recommendations. (You can also optionally modify `OPENROUTER_BASE_URL`).
Without an API key, the app returns a rule-based fallback recommendation.

```bash
cp .env.example .env
```

## Run

Sync the dependencies first:

```bash
uv sync
```

Then start the application:

```bash
uv run uvicorn apps.main:app --host 0.0.0.0 --port 8000
```

The app can also be started with:

```bash
uv run python apps/main.py
```

Once running, you can access the API documentation at [http://localhost:8000/scalar](http://localhost:8000/scalar).

## Health Checks

- `/health` is the liveness endpoint used by Railway. It returns HTTP 200
  when the FastAPI service is running, and includes artifact status in the
  response body.
- `/ready` is the strict inference readiness endpoint. It returns HTTP 503
  if the model, scaler, or threshold artifact cannot be loaded.

## Smoke Check

```bash
uv run python scripts/smoke_check.py
```

The smoke check disables the external LLM call and verifies `/health`,
`/features`, and the core `/predict` flow directly.
