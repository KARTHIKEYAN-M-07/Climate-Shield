.PHONY: help install test lint format train benchmark brief cards run-api run-dashboard clean

help:
	@echo "ClimateShield AI - Makefile Commands"
	@echo "  install        Install dependencies from requirements.txt"
	@echo "  test           Run full pytest test suite"
	@echo "  train          Run model training and benchmark pipeline"
	@echo "  benchmark      Run ingest-to-display latency benchmark"
	@echo "  brief          Generate print-ready HTML station brief"
	@echo "  cards          Generate Model Card and Data Sheet markdown"
	@echo "  run-api        Start FastAPI server (localhost:8000)"
	@echo "  run-dashboard  Start Streamlit dashboard"
	@echo "  clean          Remove temporary caches and bytecode"

install:
	pip install -r requirements.txt

test:
	pytest --maxfail=1

train:
	python -m src.train

benchmark:
	python -m src.monitoring.benchmark_latency

brief:
	python -m src.cli generate-brief --station TN_CAU_TRICHY --lang en

cards:
	python -m src.cli generate-cards

run-api:
	uvicorn src.api.main:app --reload --port 8000

run-dashboard:
	streamlit run src/dashboard/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
