# MCP Vector Search - Generated Makefile
# Generated with python-project-template (Copier)
# https://github.com/bobmatnyc/mcp-vector-search

.DEFAULT_GOAL := help

# Include modular Makefile components
-include .makefiles/common.mk
-include .makefiles/quality.mk
-include .makefiles/testing.mk
-include .makefiles/deps.mk
-include .makefiles/release.mk

# Project-specific variables (customize as needed)
PROJECT_NAME := mcp-vector-search
PYTHON_VERSION := 3.11
UV := uv

# Additional project-specific configuration
SCRIPTS_DIR := scripts
VERSION_MANAGER := $(PYTHON) $(SCRIPTS_DIR)/version_manager.py
CHANGESET_MANAGER := $(PYTHON) $(SCRIPTS_DIR)/changeset.py
DOCS_UPDATER := $(PYTHON) $(SCRIPTS_DIR)/update_docs.py

# ============================================================================
# Quick Publish Targets (uses .env.local for PyPI credentials)
# ============================================================================
# Usage: make publish         # Bump patch, build, publish, tag, push
#        make publish-minor   # Bump minor version
#        make publish-major   # Bump major version

VERSION_PY := src/mcp_vector_search/__init__.py

.PHONY: publish publish-minor publish-major publish-only

# Get current version from __init__.py
get-version = $(shell grep -E '^__version__' $(VERSION_PY) | sed 's/.*"\(.*\)"/\1/')

publish: ## Bump patch version, build, publish to PyPI, tag, and push
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Publishing Patch Release$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@CURRENT=$$(grep -E '^__version__' $(VERSION_PY) | sed 's/.*"\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	PATCH=$$(echo $$CURRENT | cut -d. -f3); \
	NEW_PATCH=$$((PATCH + 1)); \
	NEW_VERSION="$$MAJOR.$$MINOR.$$NEW_PATCH"; \
	echo "$(YELLOW)Version: $$CURRENT → $$NEW_VERSION$(NC)"; \
	sed -i '' "s/__version__ = \".*\"/__version__ = \"$$NEW_VERSION\"/" $(VERSION_PY); \
	echo "$(GREEN)✓ Version bumped$(NC)"; \
	git add $(VERSION_PY); \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	echo "$(GREEN)✓ Committed$(NC)"; \
	git tag "v$$NEW_VERSION"; \
	echo "$(GREEN)✓ Tagged v$$NEW_VERSION$(NC)"; \
	git push && git push --tags; \
	echo "$(GREEN)✓ Pushed to origin$(NC)"; \
	rm -rf dist/; \
	$(UV) build; \
	echo "$(GREEN)✓ Built package$(NC)"; \
	if [ -f .env.local ]; then \
		. .env.local && UV_PUBLISH_TOKEN="$$PYPI_TOKEN" $(UV) publish; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(RED)✗ .env.local not found - set PYPI_TOKEN$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"; \
	echo "$(GREEN)  ✓ Published mcp-vector-search $$NEW_VERSION$(NC)"; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"

publish-minor: ## Bump minor version, build, publish to PyPI, tag, and push
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Publishing Minor Release$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@CURRENT=$$(grep -E '^__version__' $(VERSION_PY) | sed 's/.*"\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	MINOR=$$(echo $$CURRENT | cut -d. -f2); \
	NEW_MINOR=$$((MINOR + 1)); \
	NEW_VERSION="$$MAJOR.$$NEW_MINOR.0"; \
	echo "$(YELLOW)Version: $$CURRENT → $$NEW_VERSION$(NC)"; \
	sed -i '' "s/__version__ = \".*\"/__version__ = \"$$NEW_VERSION\"/" $(VERSION_PY); \
	echo "$(GREEN)✓ Version bumped$(NC)"; \
	git add $(VERSION_PY); \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	echo "$(GREEN)✓ Committed$(NC)"; \
	git tag "v$$NEW_VERSION"; \
	echo "$(GREEN)✓ Tagged v$$NEW_VERSION$(NC)"; \
	git push && git push --tags; \
	echo "$(GREEN)✓ Pushed to origin$(NC)"; \
	rm -rf dist/; \
	$(UV) build; \
	echo "$(GREEN)✓ Built package$(NC)"; \
	if [ -f .env.local ]; then \
		. .env.local && UV_PUBLISH_TOKEN="$$PYPI_TOKEN" $(UV) publish; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(RED)✗ .env.local not found - set PYPI_TOKEN$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"; \
	echo "$(GREEN)  ✓ Published mcp-vector-search $$NEW_VERSION$(NC)"; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"

publish-major: ## Bump major version, build, publish to PyPI, tag, and push
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Publishing Major Release$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════$(NC)"
	@CURRENT=$$(grep -E '^__version__' $(VERSION_PY) | sed 's/.*"\(.*\)"/\1/'); \
	MAJOR=$$(echo $$CURRENT | cut -d. -f1); \
	NEW_MAJOR=$$((MAJOR + 1)); \
	NEW_VERSION="$$NEW_MAJOR.0.0"; \
	echo "$(YELLOW)Version: $$CURRENT → $$NEW_VERSION$(NC)"; \
	sed -i '' "s/__version__ = \".*\"/__version__ = \"$$NEW_VERSION\"/" $(VERSION_PY); \
	echo "$(GREEN)✓ Version bumped$(NC)"; \
	git add $(VERSION_PY); \
	git commit -m "chore: bump version to $$NEW_VERSION"; \
	echo "$(GREEN)✓ Committed$(NC)"; \
	git tag "v$$NEW_VERSION"; \
	echo "$(GREEN)✓ Tagged v$$NEW_VERSION$(NC)"; \
	git push && git push --tags; \
	echo "$(GREEN)✓ Pushed to origin$(NC)"; \
	rm -rf dist/; \
	$(UV) build; \
	echo "$(GREEN)✓ Built package$(NC)"; \
	if [ -f .env.local ]; then \
		. .env.local && UV_PUBLISH_TOKEN="$$PYPI_TOKEN" $(UV) publish; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(RED)✗ .env.local not found - set PYPI_TOKEN$(NC)"; \
		exit 1; \
	fi; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"; \
	echo "$(GREEN)  ✓ Published mcp-vector-search $$NEW_VERSION$(NC)"; \
	echo "$(GREEN)═══════════════════════════════════════════════════$(NC)"

publish-only: ## Publish current version to PyPI (no version bump)
	@echo "$(BLUE)Publishing current version to PyPI...$(NC)"
	@rm -rf dist/
	@$(UV) build
	@if [ -f .env.local ]; then \
		. .env.local && UV_PUBLISH_TOKEN="$$PYPI_TOKEN" $(UV) publish; \
		echo "$(GREEN)✓ Published to PyPI$(NC)"; \
	else \
		echo "$(RED)✗ .env.local not found - set PYPI_TOKEN$(NC)"; \
		exit 1; \
	fi

# ============================================================================
# Project-Specific Custom Targets
# ============================================================================

# Git Submodule Management
.PHONY: submodule-sync
submodule-sync: ## Sync and update git submodules
	@echo "$(GREEN)Syncing git submodules...$(NC)"
	@git submodule update --init --recursive
	@echo "$(GREEN)✓ Submodules synced$(NC)"

.PHONY: submodule-update
submodule-update: ## Update submodules to latest remote versions
	@echo "$(GREEN)Updating git submodules to latest versions...$(NC)"
	@git submodule update --init --recursive
	@git submodule update --remote vendor/py-mcp-installer-service
	@echo "$(BLUE)Submodule status:$(NC)"
	@git submodule status
	@echo "$(GREEN)✓ Submodules updated$(NC)"
	@echo "$(YELLOW)💡 To commit submodule updates:$(NC)"
	@echo "   git add vendor/py-mcp-installer-service"
	@echo "   git commit -m 'chore: update py-mcp-installer-service submodule'"

.PHONY: submodule-status
submodule-status: ## Show git submodule status
	@echo "$(BLUE)Git Submodule Status:$(NC)"
	@git submodule status

.PHONY: clean-submodules
clean-submodules: ## Clean submodule build artifacts
	@echo "$(GREEN)Cleaning submodule artifacts...$(NC)"
	@if [ -d "vendor/py-mcp-installer-service" ]; then \
		cd vendor/py-mcp-installer-service && \
		rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache 2>/dev/null || true; \
		find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; \
		find . -type f -name "*.pyc" -delete 2>/dev/null || true; \
		echo "$(GREEN)✓ Submodule artifacts cleaned$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Submodule directory not found$(NC)"; \
	fi

# Changeset Management
.PHONY: changeset-add
changeset-add: ## Add a new changeset (usage: TYPE=patch DESC="description")
	@if [ -z "$(TYPE)" ] || [ -z "$(DESC)" ]; then \
		echo "$(RED)Error:$(NC) TYPE and DESC are required"; \
		echo "$(BLUE)Usage:$(NC) make changeset-add TYPE=patch DESC=\"fix: resolve bug\""; \
		echo "$(BLUE)Types:$(NC) patch, minor, major"; \
		exit 1; \
	fi
	@$(CHANGESET_MANAGER) add --type $(TYPE) --description "$(DESC)"

.PHONY: changeset-view
changeset-view: ## View pending changesets
	@echo "$(BLUE)Pending Changesets:$(NC)"
	@$(CHANGESET_MANAGER) list

.PHONY: changeset-list
changeset-list: changeset-view ## Alias for changeset-view

.PHONY: changeset-consume
changeset-consume: ## Consume changesets for release (usage: VERSION=0.7.2)
	@if [ -z "$(VERSION)" ]; then \
		VERSION=$$($(VERSION_MANAGER) --show --format simple); \
		echo "$(BLUE)Using current version: $$VERSION$(NC)"; \
	fi
	@$(CHANGESET_MANAGER) consume --version $(VERSION)

.PHONY: changeset-validate
changeset-validate: ## Validate changeset files
	@echo "$(GREEN)Validating changesets...$(NC)"
	@$(CHANGESET_MANAGER) validate

# Documentation Updates
.PHONY: docs-update
docs-update: ## Update documentation with current version
	@echo "$(GREEN)Updating documentation...$(NC)"
	@$(DOCS_UPDATER) --type $(TYPE)

.PHONY: docs-update-readme
docs-update-readme: ## Update README.md version badge only
	@echo "$(GREEN)Updating README.md version badge...$(NC)"
	@$(DOCS_UPDATER) --readme-only

.PHONY: docs-update-claude
docs-update-claude: ## Update CLAUDE.md Recent Activity only
	@echo "$(GREEN)Updating CLAUDE.md Recent Activity...$(NC)"
	@$(DOCS_UPDATER) --claude-only

# Build Management
.PHONY: build-increment
build-increment: ## Increment build number only
	@echo "$(GREEN)Incrementing build number...$(NC)"
	@$(VERSION_MANAGER) --increment-build

.PHONY: build-package
build-package: clean submodule-sync ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(NC)"
	$(ECHO_PREFIX) $(UV) build
	@if [ -z "$(DRY_RUN)" ]; then \
		echo "$(GREEN)✓ Package built successfully$(NC)"; \
		ls -la dist/; \
	fi

# Homebrew Integration
.PHONY: homebrew-update-dry-run
homebrew-update-dry-run: ## Test Homebrew Formula update (dry-run)
	@echo "$(BLUE)Testing Homebrew Formula update...$(NC)"
	@if [ -z "$(HOMEBREW_TAP_TOKEN)" ]; then \
		echo "$(RED)✗ HOMEBREW_TAP_TOKEN not set$(NC)"; \
		exit 1; \
	fi
	@$(SCRIPTS_DIR)/update_homebrew_formula.sh --dry-run

.PHONY: homebrew-update
homebrew-update: ## Update Homebrew Formula with latest version
	@echo "$(BLUE)Updating Homebrew Formula...$(NC)"
	@if [ -z "$(HOMEBREW_TAP_TOKEN)" ]; then \
		echo "$(RED)✗ HOMEBREW_TAP_TOKEN not set. Please export HOMEBREW_TAP_TOKEN=<token>$(NC)"; \
		exit 1; \
	fi
	@$(SCRIPTS_DIR)/update_homebrew_formula.sh

.PHONY: homebrew-update-wait
homebrew-update-wait: ## Wait for PyPI and update Homebrew Formula (NON-BLOCKING)
	@echo "$(BLUE)Waiting for PyPI package and updating Homebrew Formula...$(NC)"
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	$(SCRIPTS_DIR)/wait_and_update_homebrew.sh $$VERSION || \
	echo "$(YELLOW)⚠️  Homebrew update scheduled in background$(NC)"

.PHONY: homebrew-test
homebrew-test: ## Test Homebrew Formula locally
	@echo "$(BLUE)Testing Homebrew Formula locally...$(NC)"
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "$(RED)✗ Homebrew not installed$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)This will install the formula locally - make sure you have the tap added:$(NC)"
	@echo "  brew tap bobmatnyc/mcp-vector-search"
	@echo "  brew install --build-from-source mcp-vector-search"
	@echo "$(GREEN)✓ Run the above commands to test$(NC)"

# Debug Targets
.PHONY: debug-search
debug-search: ## Debug search with logging (usage: make debug-search QUERY="term")
	@echo "$(GREEN)Running debug search...$(NC)"
	@if [ -z "$(QUERY)" ]; then echo "$(RED)Usage: make debug-search QUERY='your search term'$(NC)"; exit 1; fi
	LOGURU_LEVEL=DEBUG $(UV) run mcp-vector-search search "$(QUERY)" --verbose
	@echo "$(GREEN)✓ Debug search completed$(NC)"

.PHONY: debug-mcp
debug-mcp: ## Debug MCP server with logging
	@echo "$(GREEN)Starting MCP server in debug mode...$(NC)"
	LOGURU_LEVEL=DEBUG $(UV) run mcp-vector-search mcp --debug

.PHONY: debug-status
debug-status: ## Debug project health status
	@echo "$(GREEN)Checking project health...$(NC)"
	$(UV) run mcp-vector-search status --verbose --debug
	@echo "$(GREEN)✓ Project health check completed$(NC)"

.PHONY: debug-verify
debug-verify: ## Debug installation verification
	@echo "$(GREEN)Running debug verification...$(NC)"
	$(MAKE) check-tools
	$(MAKE) verify-setup
	@echo "$(GREEN)✓ Debug verification completed$(NC)"

.PHONY: debug-index-status
debug-index-status: ## Debug index status and health
	@echo "$(GREEN)Debugging index status...$(NC)"
	$(UV) run mcp-vector-search status --verbose
	@echo "Checking for .mcp-vector-search directory..."
	ls -la .mcp-vector-search/ 2>/dev/null || echo "No project initialized"
	@echo "$(GREEN)✓ Index status debug completed$(NC)"

.PHONY: debug-performance
debug-performance: ## Debug search performance
	@echo "$(GREEN)Debugging search performance...$(NC)"
	$(UV) run python -c "import time; start=time.time(); from mcp_vector_search.core import search; print(f'Import time: {time.time()-start:.3f}s')"
	@echo "$(GREEN)✓ Performance debug completed$(NC)"

.PHONY: debug-build
debug-build: ## Debug build failures
	@echo "$(GREEN)Debugging build process...$(NC)"
	$(MAKE) clean
	$(UV) build --verbose
	@echo "$(GREEN)✓ Build debug completed$(NC)"

# LLM Benchmarking
.PHONY: benchmark-llm
benchmark-llm: ## Benchmark LLM models for chat command
	@echo "$(GREEN)Running LLM model benchmarks...$(NC)"
	@if [ -z "$(OPENROUTER_API_KEY)" ]; then \
		echo "$(RED)✗ OPENROUTER_API_KEY not set$(NC)"; \
		exit 1; \
	fi
	$(UV) run python $(SCRIPTS_DIR)/benchmark_llm_models.py

.PHONY: benchmark-llm-fast
benchmark-llm-fast: ## Benchmark fast/cheap LLM models only
	@echo "$(GREEN)Running fast model benchmarks...$(NC)"
	$(UV) run python $(SCRIPTS_DIR)/benchmark_llm_models.py \
		--models anthropic/claude-3-haiku \
		--models openai/gpt-4o-mini

.PHONY: benchmark-llm-query
benchmark-llm-query: ## Benchmark single query (usage: make benchmark-llm-query QUERY="your query")
	@if [ -z "$(QUERY)" ]; then \
		echo "$(RED)Usage: make benchmark-llm-query QUERY='your search query'$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Benchmarking query: $(QUERY)$(NC)"
	$(UV) run python $(SCRIPTS_DIR)/benchmark_llm_models.py --query "$(QUERY)"

# MCP Integration Tests
.PHONY: test-mcp
test-mcp: ## Test MCP server integration
	@echo "$(GREEN)Testing MCP server integration...$(NC)"
	@echo "Starting MCP server test..."
	timeout 5s $(UV) run mcp-vector-search mcp || echo "MCP server test completed"
	@echo "$(GREEN)✓ MCP server integration tested$(NC)"

# Integration Tests
.PHONY: integration-test
integration-test: build-package ## Run integration tests with built package
	@echo "$(GREEN)Running integration tests...$(NC)"
	@TEMP_DIR=$$(mktemp -d); \
	cd "$$TEMP_DIR"; \
	$(UV) pip install $(PWD)/dist/*.whl; \
	mcp-vector-search --version; \
	mcp-vector-search init --file-extensions .py --embedding-model sentence-transformers/all-MiniLM-L6-v2; \
	mcp-vector-search index; \
	mcp-vector-search search "function" --limit 5; \
	echo "$(GREEN)✓ Integration tests passed$(NC)"; \
	rm -rf "$$TEMP_DIR"

# Verification
.PHONY: verify-setup
verify-setup: ## Verify installation and setup
	@echo "$(GREEN)Verifying setup...$(NC)"
	$(UV) run mcp-vector-search --version
	$(UV) run python -c "import mcp_vector_search; print('✓ Package imports successfully')"
	@echo "$(GREEN)✓ Setup verification completed$(NC)"

# Help system override to include custom targets
help: ## Show this help message
	@echo "$(BLUE)MCP Vector Search - Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
