.DEFAULT_GOAL := help
.PHONY: docs-api-build docs-api-serve \
	install-dev init init-dev lint test help test-coverage

docs-api-build: ## Build the API reference
	cd ./docs/ && make html

docs-api-serve: docs-api-build ## Serve the API reference on localhost:8071
	cd ./docs/_build/html/ && python -m http.server 8071

install-dev: ## Dev install
	python setup.py develop

init: ## Install dependencies
	pip install -r requirements.txt

init-dev: ## Install dev dependencies
	pip install -r requirements-dev.txt

lint: ## Lint code for PEP8 and PEP257 conformity
	pycodestyle .
	pydocstyle --convention=numpy .

test: ## Run unit tests
	pytest -v

test-coverage: ## Run unit tests with coverage
	coverage run --source=controlboros pytest -v

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
