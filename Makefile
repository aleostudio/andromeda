.PHONY: help setup dev test clean

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Full setup: install deps and download models
	uv sync
	uv run python setup.py
	@echo ""
	@echo "Setup complete!"

dev: ## Start Andromeda
	uv run python -m andromeda.main

test: ## Run tests
	uv run --extra dev pytest tests/ -v

clean: ## Clean caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true