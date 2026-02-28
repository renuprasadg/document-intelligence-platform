.PHONY: help install test lint format clean run

help:
    @echo "GuardianRAG Development Commands"
    @echo "================================"
    @echo "make install  - Install dependencies"
    @echo "make test     - Run tests with coverage"
    @echo "make lint     - Run code quality checks"
    @echo "make format   - Format code with black"
    @echo "make clean    - Remove cache and build files"
    @echo "make run      - Run the application"

install:
    pip install -e ".[dev]"

test:
    pytest tests/ --cov=src --cov-report=term-missing

lint:
    ruff check src/
    black --check src/

format:
    black src/
    ruff check src/ --fix

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    rm -rf .pytest_cache
    rm -rf htmlcov
    rm -rf .coverage

run:
    uvicorn knowledge_engine.main:app --reload
