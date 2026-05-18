.PHONY: help sync run dev smoke health features scalar predict-sample check

HOST ?= 0.0.0.0
PORT ?= 8000
BASE_URL ?= http://127.0.0.1:$(PORT)

help:
	@echo "DiaLens commands:"
	@echo "  make sync            Install/update dependencies with uv"
	@echo "  make run             Run API at http://$(HOST):$(PORT)"
	@echo "  make dev             Run API with auto-reload"
	@echo "  make smoke           Run direct end-to-end smoke check"
	@echo "  make health          Call /health on a running server"
	@echo "  make features        Call /features on a running server"
	@echo "  make scalar          Show Scalar API docs URL"
	@echo "  make predict-sample  Call /predict with sample payload"
	@echo "  make check           Compile Python files and validate dependencies"

sync:
	uv sync

run:
	uv run uvicorn apps.main:app --host $(HOST) --port $(PORT)

dev:
	uv run uvicorn apps.main:app --host $(HOST) --port $(PORT) --reload

smoke:
	OPENROUTER_API_KEY= uv run python scripts/smoke_check.py

health:
	curl -sS $(BASE_URL)/health

features:
	curl -sS $(BASE_URL)/features

scalar:
	@echo "$(BASE_URL)/scalar"

predict-sample:
	curl -sS -X POST $(BASE_URL)/predict \
		-H 'Content-Type: application/json' \
		-d '{"HighBP":1,"GenHlth":3,"HighChol":1,"Age":8,"CholCheck":1,"HvyAlcoholConsump":0,"BMI":28.5,"PhysActivity":1,"Smoker":0}'

check:
	uv run python -m compileall apps scripts
	uv lock --check
	uv pip check
