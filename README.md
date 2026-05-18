# DiaLens API

FastAPI service for diabetes risk screening with TensorFlow inference,
feature scaling, explainability, and Indonesian health recommendations.

## Run

```bash
uv sync
uv run uvicorn apps.main:app --host 0.0.0.0 --port 8000
```

The app can also be started with:

```bash
uv run python apps/main.py
```

## Smoke Check

```bash
uv run python scripts/smoke_check.py
```

The smoke check disables the external LLM call and verifies `/health`,
`/features`, and the core `/predict` flow directly.

## Environment

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` if you want
LLM-generated recommendations. Without an API key, the app returns a
rule-based fallback recommendation.
