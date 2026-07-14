.PHONY: install dev test lint clean build

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

bench:
	python ccia-bench/run_benchmark.py

bench-generate:
	python ccia-bench/generate_pairs.py

demo-gif:
	vhs docs/demo.tape

lint:
	python -m ruff check ctxwitch/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build:
	python -m build
