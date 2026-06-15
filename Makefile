# atlas-gateway — build system (single source of truth).
# Developers and CI run the same targets. Logic lives in scripts/, not here and
# not in the pipeline YAML. See atlas-docs/07-build-system.md.
.DEFAULT_GOAL := help
SHELL := bash

.PHONY: help lint test coverage build docker infra security ci local

help: ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n",$$1,$$2}'

lint: ## Dep age audit + Trunk(ruff lint+format) + pyright (strict)
	@./scripts/lint.sh

test: ## Offline unit tests (pytest; MockProvider + fakeredis)
	@./scripts/test.sh

coverage: ## Coverage gate (pytest-cov; recommended, ATLAS_COV_MIN)
	@./scripts/coverage.sh

build: ## Build verification + OpenAPI contract export/drift check
	@./scripts/build.sh

docker: ## Build the container image (if a Dockerfile exists)
	@./scripts/docker.sh

infra: ## Validate the deploy/ Helm chart (helm lint + template)
	@./scripts/infra.sh

security: ## Security scans (secret/CVE/fs; advisory, ATLAS_SECURITY_STRICT=1)
	@./scripts/security.sh

ci: ## Run the full gate — what CI runs
	@./scripts/ci.sh

local: ## Run the service locally (MockProvider)
	@./scripts/local.sh
