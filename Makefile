.PHONY: help install test lint format license-check license-report setup dev-setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -e ".[dev]"
	pip install pip-licenses license-expression

setup: ## Full development environment setup
	@echo "🔧 Setting up development environment..."
	python -m venv venv --prompt repo-qa || echo "Virtual environment already exists"
	@echo "✅ Virtual environment ready"
	@echo "🔄 Activating virtual environment..."
	if [ -f venv/bin/activate ]; then . venv/bin/activate; else . venv/Scripts/activate; fi
	@echo "✅ Virtual environment activated"
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Base dependencies installed"
	@echo "📦 Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"
	@echo "✅ Setup completed successfully."

dev-setup: setup ## Alias for setup command

test: ## Run all tests
	python -m pytest

test-unit: ## Run unit tests only
	python -m pytest -m "not slow and not integration"

test-integration: ## Run integration tests only
	python -m pytest -m integration

test-cov: ## Run tests with coverage report
	python -m pytest --cov=repoqa --cov-report=term-missing --cov-report=html

test-fast: ## Run tests excluding slow ones
	python -m pytest -m "not slow"

test-verbose: ## Run tests with verbose output
	python -m pytest -v

test-watch: ## Run tests in watch mode (requires pytest-watch)
	python -m pytest --tb=short --quiet

lint: ## Run linting
	python -m flake8 repoqa/
	python -m mypy repoqa/

format: ## Format code
	python -m black repoqa/
	python -m isort repoqa/

license-check: ## Check license consistency (exit code 1 if issues found)
	@./scripts/check-licenses.sh

license-report: ## Generate license report (no exit code)
	@echo "📋 License Report:"
	@python -m repoqa.license_checker --format report

license-json: ## Generate license report in JSON format
	@python -m repoqa.license_checker --format json

generate-notice: ## Generate/update NOTICE file for attributions
	@python -m repoqa.license_checker --generate-notice
	@echo "✅ NOTICE file updated with current dependencies"

generate-notice-direct: ## Generate NOTICE file for direct dependencies only
	@python -m repoqa.license_checker --generate-notice --direct-only
	@echo "✅ NOTICE file updated with direct dependencies only"

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f license-report.json license-report.txt