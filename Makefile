# MCP Vector Search - Comprehensive Build & Release System
# Version: 4.0.3
# Build: 280

# Color codes for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
RESET := \033[0m

# Default shell
SHELL := /bin/bash

# Python interpreter
PYTHON := python3
UV := uv

# Directories
SRC_DIR := src/mcp_vector_search
TEST_DIR := tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs

# Default target
.DEFAULT_GOAL := help

# Version management script
VERSION_MANAGER := $(PYTHON) $(SCRIPTS_DIR)/version_manager.py

# Changeset and documentation scripts
CHANGESET_MANAGER := $(PYTHON) $(SCRIPTS_DIR)/changeset.py
DOCS_UPDATER := $(PYTHON) $(SCRIPTS_DIR)/update_docs.py

# Check if we're in dry-run mode
ifdef DRY_RUN
	DRY_FLAG := --dry-run
	ECHO_PREFIX := @echo "$(YELLOW)[DRY RUN]$(RESET) Would execute:"
else
	DRY_FLAG :=
	ECHO_PREFIX :=
endif

# ============================================================================
# Help & Documentation
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BLUE)║     MCP Vector Search - Build & Release System              ║$(RESET)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(GREEN)Core Development:$(RESET)"
	@grep -E '^(dev|test|lint|format|clean|install):.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Git Submodules:$(RESET)"
	@grep -E '^submodule-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Version Management:$(RESET)"
	@grep -E '^version-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Changeset Management:$(RESET)"
	@grep -E '^changeset-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@grep -E '^docs-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Build Management:$(RESET)"
	@grep -E '^build-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Release Workflow:$(RESET)"
	@grep -E '^release-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Publishing:$(RESET)"
	@grep -E '^publish.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Homebrew Integration:$(RESET)"
	@grep -E '^homebrew-.*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Options:$(RESET)"
	@echo "  DRY_RUN=1       Run in dry-run mode (no actual changes)"
	@echo ""
	@echo "$(YELLOW)Examples:$(RESET)"
	@echo "  make dev                    # Install for development"
	@echo "  make test                   # Run all tests"
	@echo "  make release-patch          # Release patch version"
	@echo "  DRY_RUN=1 make release-minor # Dry-run minor release"

# ============================================================================
# Core Development Targets
# ============================================================================

.PHONY: dev
dev: ## Install for development (uv sync)
	@echo "$(GREEN)Installing for development...$(RESET)"
	$(UV) sync
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

.PHONY: install
install: ## Install package locally
	@echo "$(GREEN)Installing package...$(RESET)"
	$(UV) pip install -e .
	@echo "$(GREEN)✓ Package installed$(RESET)"

.PHONY: test
test: ## Run full test suite
	@echo "$(GREEN)Running test suite...$(RESET)"
	$(UV) run pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "$(GREEN)✓ Tests completed$(RESET)"

.PHONY: test-quick
test-quick: ## Run quick tests (no coverage)
	@echo "$(GREEN)Running quick tests...$(RESET)"
	$(UV) run pytest $(TEST_DIR) -v -x
	@echo "$(GREEN)✓ Quick tests completed$(RESET)"

.PHONY: lint
lint: ## Run linting checks (ruff, mypy)
	@echo "$(GREEN)Running linters...$(RESET)"
	@echo "  Running ruff check..."
	$(UV) run ruff check $(SRC_DIR)
	@echo "  Running ruff format check..."
	$(UV) run ruff format --check $(SRC_DIR)
	@echo "  Running mypy..."
	$(UV) run mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Linting completed$(RESET)"

.PHONY: format
format: ## Format code (ruff format)
	@echo "$(GREEN)Formatting code...$(RESET)"
	$(UV) run ruff format $(SRC_DIR)
	$(UV) run ruff check --fix $(SRC_DIR)
	@echo "$(GREEN)✓ Code formatted$(RESET)"

.PHONY: lint-fix
lint-fix: ## Fix linting issues automatically
	@echo "$(GREEN)Fixing linting issues...$(RESET)"
	$(UV) run ruff format $(SRC_DIR)
	$(UV) run ruff check --fix $(SRC_DIR)
	@echo "$(GREEN)✓ Linting issues fixed$(RESET)"

.PHONY: security
security: ## Run security checks
	@echo "$(GREEN)Running security checks...$(RESET)"
	@echo "  Running safety check..."
	$(UV) run safety check || echo "$(YELLOW)⚠️  Safety check completed with warnings$(RESET)"
	@echo "  Running bandit scan..."
	$(UV) run bandit -r $(SRC_DIR) || echo "$(YELLOW)⚠️  Bandit scan completed with warnings$(RESET)"
	@echo "$(GREEN)✓ Security checks completed$(RESET)"

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Build artifacts cleaned$(RESET)"

.PHONY: clean-submodules
clean-submodules: ## Clean submodule build artifacts
	@echo "$(GREEN)Cleaning submodule artifacts...$(RESET)"
	@if [ -d "vendor/py-mcp-installer-service" ]; then \
		cd vendor/py-mcp-installer-service && \
		rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache 2>/dev/null || true; \
		find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true; \
		find . -type f -name "*.pyc" -delete 2>/dev/null || true; \
	fi
	@echo "$(GREEN)✓ Submodule artifacts cleaned$(RESET)"

# ============================================================================
# Git Submodule Management
# ============================================================================

.PHONY: submodule-sync
submodule-sync: ## Sync and update git submodules
	@echo "$(GREEN)Syncing git submodules...$(RESET)"
	@git submodule update --init --recursive
	@echo "$(GREEN)✓ Submodules synced$(RESET)"

.PHONY: submodule-update
submodule-update: ## Update submodules to latest remote versions
	@echo "$(GREEN)Updating git submodules to latest versions...$(RESET)"
	@git submodule update --init --recursive
	@git submodule update --remote vendor/py-mcp-installer-service
	@echo "$(BLUE)Submodule status:$(RESET)"
	@git submodule status
	@echo "$(GREEN)✓ Submodules updated$(RESET)"
	@echo "$(YELLOW)💡 To commit submodule updates:$(RESET)"
	@echo "   git add vendor/py-mcp-installer-service"
	@echo "   git commit -m 'chore: update py-mcp-installer-service submodule'"

.PHONY: submodule-status
submodule-status: ## Show git submodule status
	@echo "$(BLUE)Git Submodule Status:$(RESET)"
	@git submodule status

# ============================================================================
# Version Management Targets
# ============================================================================

.PHONY: version-show
version-show: ## Display current version
	@$(VERSION_MANAGER) --show

.PHONY: version-patch
version-patch: ## Bump patch version (0.4.0 → 0.4.1)
	@echo "$(GREEN)Bumping patch version...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		$(VERSION_MANAGER) --bump patch; \
		echo "$(GREEN)✓ Patch version bumped$(RESET)"; \
	else \
		echo "$(YELLOW)[DRY RUN] Would bump patch version$(RESET)"; \
		$(VERSION_MANAGER) --bump patch --dry-run; \
	fi

.PHONY: version-bump-patch
version-bump-patch: version-patch ## Alias for version-patch

.PHONY: version-minor
version-minor: ## Bump minor version (4.0.3 → 4.1.0)
	@echo "$(GREEN)Bumping minor version...$(RESET)"
	$(ECHO_PREFIX) $(VERSION_MANAGER) --bump minor $(DRY_FLAG)
	@if [ -z "$(DRY_RUN)" ]; then \
		$(VERSION_MANAGER) --bump minor $(DRY_FLAG); \
		echo "$(GREEN)✓ Minor version bumped$(RESET)"; \
	fi

.PHONY: version-major
version-major: ## Bump major version (4.0.3 → 5.0.0)
	@echo "$(GREEN)Bumping major version...$(RESET)"
	$(ECHO_PREFIX) $(VERSION_MANAGER) --bump major $(DRY_FLAG)
	@if [ -z "$(DRY_RUN)" ]; then \
		$(VERSION_MANAGER) --bump major $(DRY_FLAG); \
		echo "$(GREEN)✓ Major version bumped$(RESET)"; \
	fi

# ============================================================================
# Changeset Management Targets
# ============================================================================

.PHONY: changeset-add
changeset-add: ## Add a new changeset (usage: TYPE=patch DESC="description")
	@if [ -z "$(TYPE)" ] || [ -z "$(DESC)" ]; then \
		echo "$(RED)Error:$(RESET) TYPE and DESC are required"; \
		echo "$(BLUE)Usage:$(RESET) make changeset-add TYPE=patch DESC=\"fix: resolve bug\""; \
		echo "$(BLUE)Types:$(RESET) patch, minor, major"; \
		exit 1; \
	fi
	@echo "$(GREEN)Adding changeset...$(RESET)"
	$(CHANGESET_MANAGER) add --type $(TYPE) --description "$(DESC)"

.PHONY: changeset-view
changeset-view: ## View pending changesets
	@echo "$(BLUE)Pending Changesets:$(RESET)"
	@$(CHANGESET_MANAGER) list

.PHONY: changeset-list
changeset-list: changeset-view ## Alias for changeset-view

.PHONY: changeset-consume
changeset-consume: ## Consume changesets for release (usage: VERSION=0.7.2)
	@if [ -z "$(VERSION)" ]; then \
		VERSION=$$($(VERSION_MANAGER) --show --format simple); \
		echo "$(BLUE)Using current version: $$VERSION$(RESET)"; \
	else \
		echo "$(BLUE)Using specified version: $(VERSION)$(RESET)"; \
	fi; \
	if [ -z "$(DRY_RUN)" ]; then \
		$(CHANGESET_MANAGER) consume --version $$VERSION; \
	else \
		$(CHANGESET_MANAGER) consume --version $$VERSION --dry-run; \
	fi

.PHONY: changeset-validate
changeset-validate: ## Validate changeset files
	@echo "$(GREEN)Validating changesets...$(RESET)"
	@$(CHANGESET_MANAGER) validate

# ============================================================================
# Documentation Update Targets
# ============================================================================

.PHONY: docs-update
docs-update: ## Update documentation with current version
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	echo "$(GREEN)Updating documentation to v$$VERSION...$(RESET)"; \
	if [ -z "$(DRY_RUN)" ]; then \
		$(DOCS_UPDATER) --version $$VERSION --type $(if $(TYPE),$(TYPE),patch); \
	else \
		$(DOCS_UPDATER) --version $$VERSION --type $(if $(TYPE),$(TYPE),patch) --dry-run; \
	fi

.PHONY: docs-update-readme
docs-update-readme: ## Update README.md version badge only
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	echo "$(GREEN)Updating README.md to v$$VERSION...$(RESET)"; \
	$(DOCS_UPDATER) --version $$VERSION --readme-only $(if $(DRY_RUN),--dry-run,)

.PHONY: docs-update-claude
docs-update-claude: ## Update CLAUDE.md Recent Activity only
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	echo "$(GREEN)Updating CLAUDE.md to v$$VERSION...$(RESET)"; \
	$(DOCS_UPDATER) --version $$VERSION --claude-only --type $(if $(TYPE),$(TYPE),patch) $(if $(DRY_RUN),--dry-run,)

# ============================================================================
# Build Management Targets
# ============================================================================

.PHONY: build-increment
build-increment: ## Increment build number only
	@echo "$(GREEN)Incrementing build number...$(RESET)"
	$(ECHO_PREFIX) $(VERSION_MANAGER) --increment-build $(DRY_FLAG)
	@if [ -z "$(DRY_RUN)" ]; then \
		$(VERSION_MANAGER) --increment-build $(DRY_FLAG); \
		echo "$(GREEN)✓ Build number incremented$(RESET)"; \
	fi

.PHONY: build-package
build-package: clean submodule-sync ## Build distribution packages
	@echo "$(GREEN)Building distribution packages...$(RESET)"
	$(ECHO_PREFIX) $(UV) build
	@if [ -z "$(DRY_RUN)" ]; then \
		$(UV) build; \
		echo "$(GREEN)✓ Package built$(RESET)"; \
	fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

.PHONY: preflight-check
preflight-check:
	@echo "$(BLUE)Running pre-flight checks...$(RESET)"
	@# Check and sync git submodules
	@echo "$(BLUE)Checking git submodules...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		$(MAKE) submodule-sync; \
		if [ -d "vendor/py-mcp-installer-service/.git" ]; then \
			echo "$(GREEN)✓ Git submodules synced$(RESET)"; \
		else \
			echo "$(RED)✗ Submodule sync failed$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)[DRY RUN] Skipping submodule sync$(RESET)"; \
	fi
	@# Check git status
	@if [ -z "$(DRY_RUN)" ]; then \
		if [ -n "$$(git status --porcelain)" ]; then \
			echo "$(RED)✗ Git working directory is not clean$(RESET)"; \
			echo "  Please commit or stash your changes first."; \
			exit 1; \
		else \
			echo "$(GREEN)✓ Git working directory is clean$(RESET)"; \
		fi; \
	else \
		echo "$(YELLOW)[DRY RUN] Skipping git status check$(RESET)"; \
	fi
	@# Check tests pass
	@echo "$(BLUE)Running tests...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		if $(UV) run pytest $(TEST_DIR) -q; then \
			echo "$(GREEN)✓ All tests pass$(RESET)"; \
		else \
			echo "$(RED)✗ Tests failed$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)[DRY RUN] Skipping test execution$(RESET)"; \
	fi
	@# Check linting
	@echo "$(BLUE)Checking code quality...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		if $(UV) run ruff check $(SRC_DIR) --quiet; then \
			echo "$(GREEN)✓ Code quality checks pass$(RESET)"; \
		else \
			echo "$(RED)✗ Linting issues found$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)[DRY RUN] Skipping linting check$(RESET)"; \
	fi

# ============================================================================
# Release Workflow Targets
# ============================================================================

.PHONY: release-patch
release-patch: preflight-check ## Full release with patch bump
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Starting PATCH release workflow$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(MAKE) version-patch
	$(MAKE) build-increment
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	$(MAKE) changeset-consume VERSION=$$VERSION
	$(MAKE) docs-update TYPE=patch
	$(MAKE) git-commit-release
	$(MAKE) build-package
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  PATCH release completed successfully!$(RESET)"
	@echo "$(GREEN)  Next: Run 'make publish' to upload to PyPI$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

.PHONY: release-minor
release-minor: preflight-check ## Full release with minor bump
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Starting MINOR release workflow$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(MAKE) version-minor
	$(MAKE) build-increment
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	$(MAKE) changeset-consume VERSION=$$VERSION
	$(MAKE) docs-update TYPE=minor
	$(MAKE) git-commit-release
	$(MAKE) build-package
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  MINOR release completed successfully!$(RESET)"
	@echo "$(GREEN)  Next: Run 'make publish' to upload to PyPI$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

.PHONY: release-major
release-major: preflight-check ## Full release with major bump
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Starting MAJOR release workflow$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(MAKE) version-major
	$(MAKE) build-increment
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	$(MAKE) changeset-consume VERSION=$$VERSION
	$(MAKE) docs-update TYPE=major
	$(MAKE) git-commit-release
	$(MAKE) build-package
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  MAJOR release completed successfully!$(RESET)"
	@echo "$(GREEN)  Next: Run 'make publish' to upload to PyPI$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

# ============================================================================
# Git Operations
# ============================================================================

.PHONY: git-commit-release
git-commit-release:
	@echo "$(GREEN)Committing release changes...$(RESET)"
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	if [ -z "$(DRY_RUN)" ]; then \
		git add -A; \
		git commit -m "🚀 Release v$$VERSION"; \
		git tag -a "v$$VERSION" -m "Release version $$VERSION"; \
		echo "$(GREEN)✓ Release committed and tagged as v$$VERSION$(RESET)"; \
	else \
		echo "$(YELLOW)[DRY RUN] Would commit with message: Release v$$VERSION$(RESET)"; \
		echo "$(YELLOW)[DRY RUN] Would create tag: v$$VERSION$(RESET)"; \
	fi

.PHONY: git-push
git-push: ## Push commits and tags to origin
	@echo "$(GREEN)Pushing to origin...$(RESET)"
	$(ECHO_PREFIX) git push origin main --tags
	@if [ -z "$(DRY_RUN)" ]; then \
		git push origin main --tags; \
		echo "$(GREEN)✓ Pushed to origin$(RESET)"; \
	fi

.PHONY: git-status
git-status: ## Show git status and recent commits
	@echo "$(BLUE)Git Status:$(RESET)"
	@git status --short
	@echo "\n$(BLUE)Recent Commits:$(RESET)"
	@git log --oneline -10

.PHONY: git-clean
git-clean: ## Clean git repository (remove untracked files)
	@echo "$(YELLOW)⚠️  This will remove all untracked files!$(RESET)"
	@read -p "Are you sure? (y/N): " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		git clean -fd; \
		echo "$(GREEN)✓ Repository cleaned$(RESET)"; \
	else \
		echo "$(BLUE)Cancelled$(RESET)"; \
	fi

# ============================================================================
# Changelog Management
# ============================================================================

.PHONY: changelog-update
changelog-update:
	@echo "$(GREEN)Updating changelog...$(RESET)"
	$(ECHO_PREFIX) $(VERSION_MANAGER) --update-changelog $(DRY_FLAG)
	@if [ -z "$(DRY_RUN)" ]; then \
		$(VERSION_MANAGER) --update-changelog $(DRY_FLAG); \
		echo "$(GREEN)✓ Changelog updated$(RESET)"; \
	fi

# ============================================================================
# Publishing Targets
# ============================================================================

.PHONY: publish
publish: ## Publish to PyPI
	@echo "$(BLUE)Publishing to PyPI...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		if [ ! -d "dist" ]; then \
			echo "$(RED)✗ No dist/ directory found. Run 'make build-package' first.$(RESET)"; \
			exit 1; \
		fi; \
		$(UV) publish; \
		echo "$(GREEN)✓ Published to PyPI$(RESET)"; \
	else \
		echo "$(YELLOW)[DRY RUN] Would publish to PyPI$(RESET)"; \
	fi

.PHONY: publish-test
publish-test: ## Publish to test PyPI
	@echo "$(BLUE)Publishing to Test PyPI...$(RESET)"
	@if [ -z "$(DRY_RUN)" ]; then \
		if [ ! -d "dist" ]; then \
			echo "$(RED)✗ No dist/ directory found. Run 'make build-package' first.$(RESET)"; \
			exit 1; \
		fi; \
		$(UV) publish --publish-url https://test.pypi.org/legacy/; \
		echo "$(GREEN)✓ Published to Test PyPI$(RESET)"; \
	else \
		echo "$(YELLOW)[DRY RUN] Would publish to Test PyPI$(RESET)"; \
	fi

.PHONY: homebrew-update-dry-run
homebrew-update-dry-run: ## Test Homebrew Formula update (dry-run)
	@echo "$(BLUE)Testing Homebrew Formula update...$(RESET)"
	@if [ -z "$(HOMEBREW_TAP_TOKEN)" ]; then \
		echo "$(RED)✗ HOMEBREW_TAP_TOKEN not set$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/update_homebrew_formula.py --dry-run --verbose

.PHONY: homebrew-update
homebrew-update: ## Update Homebrew Formula with latest version
	@echo "$(BLUE)Updating Homebrew Formula...$(RESET)"
	@if [ -z "$(HOMEBREW_TAP_TOKEN)" ]; then \
		echo "$(RED)✗ HOMEBREW_TAP_TOKEN not set. Please export HOMEBREW_TAP_TOKEN=<token>$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/update_homebrew_formula.py --verbose
	@echo "$(GREEN)✓ Homebrew Formula updated$(RESET)"

.PHONY: homebrew-update-wait
homebrew-update-wait: ## Wait for PyPI and update Homebrew Formula (NON-BLOCKING)
	@echo "$(BLUE)Waiting for PyPI package and updating Homebrew Formula...$(RESET)"
	@VERSION=$$($(VERSION_MANAGER) --show --format simple); \
	$(SCRIPTS_DIR)/wait_and_update_homebrew.sh $$VERSION || \
	(echo "$(YELLOW)⚠️  Homebrew update failed (NON-BLOCKING)$(RESET)"; exit 0)

.PHONY: homebrew-test
homebrew-test: ## Test Homebrew Formula locally
	@echo "$(BLUE)Testing Homebrew Formula locally...$(RESET)"
	@if ! command -v brew >/dev/null 2>&1; then \
		echo "$(RED)✗ Homebrew not installed$(RESET)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)This will install the formula locally - make sure you have the tap added:$(RESET)"
	@echo "  brew tap bobmatnyc/mcp-vector-search"
	@echo "  brew install --build-from-source mcp-vector-search"
	@echo "$(GREEN)✓ Run the above commands to test$(RESET)"

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: check-tools
check-tools: ## Check required tools are installed
	@echo "$(BLUE)Checking required tools...$(RESET)"
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)✗ Python not found$(RESET)"; exit 1; }
	@echo "$(GREEN)✓ Python found: $$($(PYTHON) --version)$(RESET)"
	@command -v $(UV) >/dev/null 2>&1 || { echo "$(RED)✗ uv not found$(RESET)"; exit 1; }
	@echo "$(GREEN)✓ uv found: $$($(UV) --version)$(RESET)"
	@command -v git >/dev/null 2>&1 || { echo "$(RED)✗ git not found$(RESET)"; exit 1; }
	@echo "$(GREEN)✓ git found: $$(git --version)$(RESET)"

.PHONY: info
info: ## Show project information
	@echo "$(BLUE)Project Information:$(RESET)"
	@$(VERSION_MANAGER) --show --format detailed

.PHONY: performance
performance: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	@if [ ! -f "$(SCRIPTS_DIR)/search_performance_monitor.py" ]; then \
		echo "$(RED)✗ Performance monitor script not found$(RESET)"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPTS_DIR)/search_performance_monitor.py --quality --save
	@echo "$(GREEN)✓ Performance benchmarks completed$(RESET)"

.PHONY: integration-test
integration-test: build-package ## Run integration tests with built package
	@echo "$(GREEN)Running integration tests...$(RESET)"
	@# Create temporary test environment
	@TEMP_DIR=$$(mktemp -d); \
	echo "Testing in $$TEMP_DIR"; \
	cd "$$TEMP_DIR"; \
	echo "def test_function(): pass" > test.py; \
	$(UV) pip install $(PWD)/dist/*.whl; \
	mcp-vector-search --version; \
	mcp-vector-search init --file-extensions .py --embedding-model sentence-transformers/all-MiniLM-L6-v2; \
	mcp-vector-search index; \
	mcp-vector-search search "function" --limit 5; \
	echo "$(GREEN)✓ Integration tests passed$(RESET)"; \
	rm -rf "$$TEMP_DIR"

.PHONY: full-release
full-release: preflight-check ## Complete release workflow (build, test, publish, push)
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Starting FULL release workflow$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(MAKE) release-patch
	$(MAKE) integration-test
	$(MAKE) publish
	@if [ -n "$(HOMEBREW_TAP_TOKEN)" ]; then \
		$(MAKE) homebrew-update; \
	else \
		echo "$(YELLOW)⚠️  Skipping Homebrew update (HOMEBREW_TAP_TOKEN not set)$(RESET)"; \
	fi
	$(MAKE) git-push
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  FULL release completed successfully!$(RESET)"
	@echo "$(GREEN)  Package published and pushed to origin$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

.PHONY: build-comprehensive
build-comprehensive: ## Run comprehensive build (all checks, tests, build)
	@echo "$(GREEN)Running comprehensive build...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/comprehensive_build.py
	@echo "$(GREEN)✓ Comprehensive build completed$(RESET)"

.PHONY: build-quick
build-quick: ## Quick build (skip some checks for speed)
	@echo "$(GREEN)Running quick build...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/comprehensive_build.py --skip-security --skip-integration --no-coverage
	@echo "$(GREEN)✓ Quick build completed$(RESET)"

.PHONY: build-fix
build-fix: ## Build with automatic linting fixes
	@echo "$(GREEN)Running build with fixes...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/comprehensive_build.py --fix-lint
	@echo "$(GREEN)✓ Build with fixes completed$(RESET)"

.PHONY: setup-dev
setup-dev: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	$(UV) sync --dev
	$(UV) run pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(RESET)"

.PHONY: ci-check
ci-check: ## Run CI-like checks locally
	@echo "$(GREEN)Running CI checks...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/comprehensive_build.py --skip-setup
	@echo "$(GREEN)✓ CI checks completed$(RESET)"

# ============================================================================
# Single-Path Workflows (Claude Code Optimization)
# ============================================================================

.PHONY: dev-setup
dev-setup: check-tools ## 🔴 ONE-COMMAND development environment setup
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Setting up MCP Vector Search development environment$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(UV) sync --dev
	$(UV) pip install -e .
	$(UV) run pre-commit install
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  Development environment ready!$(RESET)"
	@echo "$(GREEN)  Next: make test (run tests)$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

.PHONY: quality
quality: ## 🔴 Run ALL quality checks (lint, type, security, test)
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  Running comprehensive quality checks$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════$(RESET)"
	$(MAKE) lint
	$(MAKE) security
	$(MAKE) test
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  All quality checks passed!$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════$(RESET)"

.PHONY: typecheck
typecheck: ## 🔴 Type checking with mypy
	@echo "$(GREEN)Running type checking...$(RESET)"
	$(UV) run mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking completed$(RESET)"

.PHONY: build
build: clean ## 🔴 Build package for distribution
	@echo "$(GREEN)Building package...$(RESET)"
	$(UV) build
	@echo "$(GREEN)✓ Package built successfully$(RESET)"

.PHONY: test-unit
test-unit: ## 🔴 Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	$(UV) run pytest $(TEST_DIR) -v -m "not integration" --cov=$(SRC_DIR) --cov-report=term-missing
	@echo "$(GREEN)✓ Unit tests completed$(RESET)"

.PHONY: test-integration
test-integration: ## 🔴 Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	$(UV) run pytest $(TEST_DIR) -v -m "integration"
	@echo "$(GREEN)✓ Integration tests completed$(RESET)"

.PHONY: test-mcp
test-mcp: ## 🔴 Test MCP server integration
	@echo "$(GREEN)Testing MCP server integration...$(RESET)"
	@echo "Starting MCP server test..."
	timeout 5s $(UV) run mcp-vector-search mcp || echo "MCP server test completed"
	@echo "$(GREEN)✓ MCP server integration tested$(RESET)"

.PHONY: verify-setup
verify-setup: ## 🔴 Verify installation and setup
	@echo "$(GREEN)Verifying setup...$(RESET)"
	$(UV) run mcp-vector-search --version
	$(UV) run python -c "import mcp_vector_search; print('✓ Package imports successfully')"
	@echo "$(GREEN)✓ Setup verification completed$(RESET)"

.PHONY: version-check
version-check: ## 🔴 Validate version consistency
	@echo "$(GREEN)Checking version consistency...$(RESET)"
	$(VERSION_MANAGER) --validate
	@echo "$(GREEN)✓ Version consistency validated$(RESET)"

# ============================================================================
# Debug Commands (Referenced in CLAUDE.md)
# ============================================================================

.PHONY: debug-search
debug-search: ## 🟡 Debug search with logging (usage: make debug-search QUERY="term")
	@echo "$(GREEN)Running debug search...$(RESET)"
	@if [ -z "$(QUERY)" ]; then echo "$(RED)Usage: make debug-search QUERY='your search term'$(RESET)"; exit 1; fi
	LOGURU_LEVEL=DEBUG $(UV) run mcp-vector-search search "$(QUERY)" --verbose
	@echo "$(GREEN)✓ Debug search completed$(RESET)"

.PHONY: debug-mcp
debug-mcp: ## 🟡 Debug MCP server with logging
	@echo "$(GREEN)Starting MCP server in debug mode...$(RESET)"
	LOGURU_LEVEL=DEBUG $(UV) run mcp-vector-search mcp --debug

.PHONY: debug-status
debug-status: ## 🟡 Debug project health status
	@echo "$(GREEN)Checking project health...$(RESET)"
	$(UV) run mcp-vector-search status --verbose --debug
	@echo "$(GREEN)✓ Project health check completed$(RESET)"

.PHONY: debug-verify
debug-verify: ## 🟡 Debug installation verification
	@echo "$(GREEN)Running debug verification...$(RESET)"
	$(MAKE) check-tools
	$(MAKE) verify-setup
	@echo "$(GREEN)✓ Debug verification completed$(RESET)"

# ============================================================================
# Debugging Support Commands (Referenced in CLAUDE.md)
# ============================================================================

.PHONY: debug-index-status
debug-index-status: ## 🟢 Debug index status and health
	@echo "$(GREEN)Debugging index status...$(RESET)"
	$(UV) run mcp-vector-search status --verbose
	@echo "Checking for .mcp-vector-search directory..."
	ls -la .mcp-vector-search/ 2>/dev/null || echo "No project initialized"
	@echo "$(GREEN)✓ Index status debug completed$(RESET)"

.PHONY: debug-performance
debug-performance: ## 🟢 Debug search performance
	@echo "$(GREEN)Debugging search performance...$(RESET)"
	$(UV) run python -c "import time; start=time.time(); from mcp_vector_search.core import search; print(f'Import time: {time.time()-start:.3f}s')"
	@echo "$(GREEN)✓ Performance debug completed$(RESET)"

.PHONY: debug-build
debug-build: ## 🟢 Debug build failures
	@echo "$(GREEN)Debugging build process...$(RESET)"
	$(MAKE) clean
	$(UV) build --verbose
	@echo "$(GREEN)✓ Build debug completed$(RESET)"

# Prevent make from treating arguments as targets
%:
	@:
