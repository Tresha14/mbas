# MBAS task runner. On Windows without `make`, use the matching targets in tasks.ps1.
ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python
else
PY ?= .venv/bin/python
endif

.PHONY: help venv install torch-cuda up down ps recorder-api recorder-web recorder-web-install questions check test

help:
	@echo "venv / install / torch-cuda  set up Python env"
	@echo "up / down / ps               docker compose (postgres, redis, livekit, caddy)"
	@echo "recorder-api                 FastAPI recorder backend on :8010"
	@echo "recorder-web-install         npm install for the recorder web app"
	@echo "recorder-web                 Next.js recorder on :3010"
	@echo "check S=<session_dir>        verify a recorded session"
	@echo "test                         run pytest"

venv:
	python -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

torch-cuda:
	$(PY) -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

up:
	docker compose up -d
	docker compose ps

down:
	docker compose down

ps:
	docker compose ps

questions:
	$(PY) -m tools.make_questions

recorder-api:
	$(PY) -m uvicorn recorder.backend.main:app --host 0.0.0.0 --port 8010 --reload --reload-dir recorder --reload-dir tools

recorder-web-install:
	cd recorder/web && npm install

recorder-web:
	cd recorder/web && npm run dev

check:
	$(PY) -m tools.check_session $(S)

test:
	$(PY) -m pytest -q tests
