.DEFAULT_GOAL := help
.PHONY: install-dev init init-dev lint test help

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

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
